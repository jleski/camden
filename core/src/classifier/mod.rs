//! AI-powered image classification for content moderation and tagging.
//!
//! This module provides ONNX-based inference for:
//! - Content moderation (NSFW detection with 5 classes)
//! - Automatic image tagging (ImageNet classification)
//!
//! # Configuration
//!
//! Models can be configured via a TOML file (`camden-classifier.toml`):
//!
//! ```toml
//! models_dir = ".vendor/models"
//! active_moderation = "gantman-nsfw"
//! active_tagging = "mobilenetv2"
//!
//! [models.gantman-nsfw]
//! name = "GantMan NSFW"
//! type = "moderation"
//! path = "nsfw-inception-v3.onnx"
//! ```
//!
//! # Runtime Initialization
//!
//! Before using any classifier, you must initialize the ONNX Runtime by calling
//! [`init_ort_runtime`] with the path to the `onnxruntime.dll`. This is required
//! because we use dynamic loading to avoid CRT conflicts with static OpenCV.
//!
//! ```no_run
//! use camden_core::classifier::init_ort_runtime;
//!
//! // Initialize once at application startup
//! init_ort_runtime(".vendor/onnxruntime/lib/onnxruntime.dll").unwrap();
//! ```

mod config;
mod ensemble;
mod labels;
mod moderation;
mod models;
mod runtime;
mod tagging;

pub use config::{ClassifierConfig, ModelConfig, ModelInputSpec, ModelOutputSpec, ModelPreset, ModelType, DEFAULT_MAX_TAGS_SINGLE, DEFAULT_MAX_TAGS_ENSEMBLE};
pub use moderation::{AggregationStrategy, EnsembleModerationClassifier, ModerationCategories, ModerationConfig, ModerationFlags, ModerationModelFormat, ModerationTier, NsfwClassifier};
pub use runtime::{ClassifierError, ModelPaths};
pub use tagging::{EnsembleTaggingClassifier, ImageTag, TagCategory, TaggingClassifier, TaggingConfig};

use std::path::Path;
use std::sync::OnceLock;

/// Global flag to track if ORT runtime has been initialized.
static ORT_INITIALIZED: OnceLock<()> = OnceLock::new();

/// Initialize the ONNX Runtime with the path to the dynamic library.
///
/// This must be called once before using any classifier. The function is
/// idempotent - subsequent calls after successful initialization are no-ops.
///
/// # Arguments
///
/// * `dylib_path` - Path to `onnxruntime.dll` (Windows) or `libonnxruntime.so` (Linux)
///
/// # Errors
///
/// Returns an error if the library cannot be loaded or initialized.
///
/// # Example
///
/// ```no_run
/// use camden_core::classifier::init_ort_runtime;
///
/// // From .vendor directory
/// init_ort_runtime(".vendor/onnxruntime/lib/onnxruntime.dll")?;
/// # Ok::<(), camden_core::classifier::ClassifierError>(())
/// ```
pub fn init_ort_runtime(dylib_path: impl AsRef<Path>) -> Result<(), ClassifierError> {
    let path = dylib_path.as_ref();
    
    if ORT_INITIALIZED.get().is_some() {
        return Ok(());
    }
    
    if !path.exists() {
        return Err(ClassifierError::Processing(format!(
            "ONNX Runtime library not found at: {}. Run 'task deps-onnxruntime' to download.",
            path.display()
        )));
    }
    
    let path_str = path.to_str().ok_or_else(|| {
        ClassifierError::Processing("ONNX Runtime path contains invalid UTF-8".to_string())
    })?;
    
    ort::init_from(path_str)
        .commit()
        .map_err(ClassifierError::Ort)?;
    
    let _ = ORT_INITIALIZED.set(());
    Ok(())
}

/// Get the default path to the ONNX Runtime library.
///
/// Returns the path relative to the workspace root (.vendor/onnxruntime/lib/onnxruntime.dll).
pub fn default_ort_dylib_path() -> std::path::PathBuf {
    #[cfg(windows)]
    {
        std::path::PathBuf::from(".vendor/onnxruntime/lib/onnxruntime.dll")
    }
    #[cfg(not(windows))]
    {
        std::path::PathBuf::from(".vendor/onnxruntime/lib/libonnxruntime.so")
    }
}

/// Moderation classifier variant (single or ensemble).
enum ModerationClassifierVariant {
    Single(NsfwClassifier),
    Ensemble(EnsembleModerationClassifier),
}

/// Tagging classifier variant (single or ensemble).
enum TaggingClassifierVariant {
    Single(TaggingClassifier),
    Ensemble(EnsembleTaggingClassifier),
}

/// Combined classifier that runs both moderation and tagging models.
pub struct ImageClassifier {
    moderation: ModerationClassifierVariant,
    tagging: TaggingClassifierVariant,
    config: ClassifierConfig,
}

/// Load moderation classifier (single or ensemble) from configuration.
///
/// Handles both single-model and ensemble modes, loading the appropriate
/// classifier variant based on the configuration.
///
/// # Arguments
///
/// * `config` - Classifier configuration with model settings
///
/// # Returns
///
/// A `ModerationClassifierVariant` (either Single or Ensemble)
fn load_moderation_classifier(
    config: &ClassifierConfig,
) -> Result<ModerationClassifierVariant, ClassifierError> {
    let models_dir = &config.models_dir;

    if config.is_ensemble_mode() {
        // Ensemble mode: load multiple models
        let model_configs: Vec<_> = config
            .active_moderation_models()
            .iter()
            .filter_map(|model_config| {
                let path = if model_config.path.is_absolute() {
                    model_config.path.clone()
                } else {
                    models_dir.join(&model_config.path)
                };
                let mod_config = ModerationConfig::from_specs(
                    &model_config.input,
                    &model_config.output.labels,
                    model_config.output.format.as_deref(),
                );
                Some((path, mod_config))
            })
            .collect();

        if model_configs.is_empty() {
            return Err(ClassifierError::Processing(
                "no valid moderation models configured for ensemble".to_string(),
            ));
        }

        Ok(ModerationClassifierVariant::Ensemble(
            EnsembleModerationClassifier::with_configs(model_configs)?,
        ))
    } else {
        // Single model mode
        let moderation_path = config.active_moderation_path().ok_or_else(|| {
            ClassifierError::Processing("no active moderation model configured".to_string())
        })?;

        let single_classifier = if let Some(model_config) = config.active_moderation_model() {
            let mod_config = ModerationConfig::from_specs(
                &model_config.input,
                &model_config.output.labels,
                model_config.output.format.as_deref(),
            );
            NsfwClassifier::with_config(&moderation_path, mod_config)?
        } else {
            NsfwClassifier::new(&moderation_path)?
        };

        Ok(ModerationClassifierVariant::Single(single_classifier))
    }
}

/// Load tagging classifier (single or ensemble) from configuration.
///
/// Handles both single-model and ensemble modes, loading the appropriate
/// classifier variant based on the configuration. Also loads label files
/// for each model.
///
/// # Arguments
///
/// * `config` - Classifier configuration with model settings
///
/// # Returns
///
/// A `TaggingClassifierVariant` (either Single or Ensemble)
fn load_tagging_classifier(
    config: &ClassifierConfig,
) -> Result<TaggingClassifierVariant, ClassifierError> {
    let models_dir = &config.models_dir;

    if config.is_tagging_ensemble_mode() {
        // Tagging ensemble mode: load multiple models
        let model_configs: Vec<_> = config
            .active_tagging_models()
            .iter()
            .filter_map(|model_config| {
                let path = if model_config.path.is_absolute() {
                    model_config.path.clone()
                } else {
                    models_dir.join(&model_config.path)
                };
                let tag_config = TaggingConfig::from_specs_with_output(
                    &model_config.input,
                    model_config.output.multi_label,
                );
                let labels = load_tagging_labels(models_dir, &model_config.output).ok()?;
                Some((path, tag_config, labels))
            })
            .collect();

        if model_configs.is_empty() {
            return Err(ClassifierError::Processing(
                "no valid tagging models configured for ensemble".to_string(),
            ));
        }

        Ok(TaggingClassifierVariant::Ensemble(
            EnsembleTaggingClassifier::with_configs(model_configs)?,
        ))
    } else {
        // Single tagging model mode
        let tagging_path = config.active_tagging_path().ok_or_else(|| {
            ClassifierError::Processing("no active tagging model configured".to_string())
        })?;

        let single_classifier = if let Some(model_config) = config.active_tagging_model() {
            let tag_config = TaggingConfig::from_specs_with_output(
                &model_config.input,
                model_config.output.multi_label,
            );
            let labels = load_tagging_labels(models_dir, &model_config.output)?;
            TaggingClassifier::with_config_and_labels(&tagging_path, tag_config, labels)?
        } else {
            TaggingClassifier::new(&tagging_path)?
        };

        Ok(TaggingClassifierVariant::Single(single_classifier))
    }
}

impl ImageClassifier {
    /// Create a new classifier with models from the specified paths.
    pub fn new(paths: &ModelPaths) -> Result<Self, ClassifierError> {
        let moderation = ModerationClassifierVariant::Single(NsfwClassifier::new(&paths.nsfw_model)?);
        let tagging = TaggingClassifierVariant::Single(TaggingClassifier::new(&paths.tagging_model)?);
        Ok(Self {
            moderation,
            tagging,
            config: ClassifierConfig::default(),
        })
    }

    /// Create a classifier using default model paths (.vendor/models/).
    pub fn with_default_paths() -> Result<Self, ClassifierError> {
        let paths = ModelPaths::default();
        Self::new(&paths)
    }
    
    /// Create a classifier from a configuration.
    pub fn from_config(config: ClassifierConfig) -> Result<Self, ClassifierError> {
        let moderation = load_moderation_classifier(&config)?;
        let tagging = load_tagging_classifier(&config)?;

        Ok(Self {
            moderation,
            tagging,
            config,
        })
    }
    
    /// Create a classifier by loading config from file or using defaults.
    pub fn from_config_or_default() -> Result<Self, ClassifierError> {
        let config = ClassifierConfig::load_or_default();
        Self::from_config(config)
    }
    
    /// Get the current configuration.
    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Analyze an image for moderation flags only.
    pub fn moderate(&mut self, image_path: &Path) -> Result<ModerationFlags, ClassifierError> {
        match &mut self.moderation {
            ModerationClassifierVariant::Single(classifier) => classifier.classify(image_path),
            ModerationClassifierVariant::Ensemble(classifier) => classifier.classify(image_path),
        }
    }

    /// Generate tags for an image.
    pub fn tag(&mut self, image_path: &Path, max_tags: usize) -> Result<Vec<ImageTag>, ClassifierError> {
        match &mut self.tagging {
            TaggingClassifierVariant::Single(classifier) => classifier.classify(image_path, max_tags),
            TaggingClassifierVariant::Ensemble(classifier) => classifier.classify(image_path, max_tags),
        }
    }

    /// Run full classification (moderation + tagging).
    ///
    /// This method runs both moderation and tagging, then cross-references
    /// the results to adjust the moderation tier based on tag evidence.
    /// This helps catch cases where explicit tags are detected but the
    /// moderation model produces a lower tier.
    pub fn classify(&mut self, image_path: &Path) -> Result<ClassificationResult, ClassifierError> {
        let moderation = match &mut self.moderation {
            ModerationClassifierVariant::Single(classifier) => classifier.classify(image_path)?,
            ModerationClassifierVariant::Ensemble(classifier) => classifier.classify(image_path)?,
        };
        let tags = match &mut self.tagging {
            TaggingClassifierVariant::Single(classifier) => classifier.classify(image_path, 10)?,
            TaggingClassifierVariant::Ensemble(classifier) => classifier.classify(image_path, 10)?,
        };

        // Adjust moderation tier based on tag evidence
        let moderation = adjust_tier_from_tags(moderation, &tags);

        Ok(ClassificationResult { moderation, tags })
    }
}

/// Complete classification result containing moderation and tags.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClassificationResult {
    pub moderation: ModerationFlags,
    pub tags: Vec<ImageTag>,
}

/// NSFW-indicative tags with their severity weights for tier adjustment.
/// Weight represents how strongly a tag indicates NSFW content (0.0-1.0).
const NSFW_TAG_WEIGHTS: &[(&str, f32)] = &[
    // Explicit content tags (highest weight)
    ("explicit", 1.0),
    ("sex", 1.0),
    ("penis", 0.95),
    ("pussy", 0.95),
    ("vaginal", 0.95),
    ("anal", 0.95),
    ("cum", 0.9),
    ("nude", 0.9),
    ("naked", 0.9),
    ("bottomless", 0.85),
    // Partial nudity tags
    ("topless", 0.7),
    ("nipples", 0.65),
    ("areola", 0.65),
    // Suggestive content tags
    ("breasts", 0.35),
    ("cleavage", 0.3),
    ("underwear", 0.25),
    ("bikini", 0.2),
    ("swimsuit", 0.15),
    ("ass", 0.25),
    ("thighs", 0.15),
];

/// Adjust moderation tier based on tag evidence.
///
/// Cross-references detected tags with NSFW-indicative keywords to potentially
/// escalate the moderation tier when models disagree. This helps catch cases
/// where tagging models detect explicit content that moderation models miss.
fn adjust_tier_from_tags(mut flags: ModerationFlags, tags: &[ImageTag]) -> ModerationFlags {
    // Calculate cumulative NSFW evidence from tags
    let nsfw_evidence: f32 = tags
        .iter()
        .filter_map(|tag| {
            let tag_lower = tag.name.to_lowercase();
            NSFW_TAG_WEIGHTS
                .iter()
                .find(|(name, _)| tag_lower.contains(name))
                .map(|(_, weight)| tag.confidence * weight)
        })
        .sum();

    // Escalate tier based on cumulative evidence
    // Thresholds are tuned to avoid false escalations while catching obvious misses
    let new_tier = match (flags.tier, nsfw_evidence) {
        // Strong evidence (explicit tags with high confidence) -> Restricted
        (_, e) if e > 1.5 => ModerationTier::Restricted,
        // Moderate-high evidence -> Mature (unless already higher)
        (ModerationTier::Safe, e) if e > 0.9 => ModerationTier::Mature,
        (ModerationTier::Sensitive, e) if e > 0.9 => ModerationTier::Mature,
        // Moderate evidence -> Sensitive (unless already higher)
        (ModerationTier::Safe, e) if e > 0.5 => ModerationTier::Sensitive,
        // Keep existing tier if no significant evidence
        (tier, _) => tier,
    };

    // Only escalate, never downgrade
    if new_tier.level() > flags.tier.level() {
        flags.tier = new_tier;
    }

    flags
}

/// Load tagging labels from model output specification.
///
/// This is a convenience wrapper around `labels::load_labels`.
fn load_tagging_labels(
    models_dir: &Path,
    output: &ModelOutputSpec,
) -> Result<Vec<String>, ClassifierError> {
    labels::load_labels(models_dir, output)
}
