//! Image tagging using configurable ONNX models (MobileNetV2, EfficientNet, ViT, etc.).
//!
//! Supports multiple ImageNet-based models with configurable input sizes and normalization.

use super::config::ModelInputSpec;
use super::runtime::{
    load_session, preprocess_image_with_layout, sigmoid, softmax, ClassifierError, IMAGE_NET_MEAN,
    IMAGE_NET_STD,
};
use once_cell::sync::Lazy;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Default minimum confidence threshold for tags (0.0-1.0).
/// Tags below this threshold are filtered out.
/// 0.35 = 35% confidence - lower threshold allows more semantic tags from
/// multi-label models (WD taggers) where probability mass is spread across 10k+ tags.
pub const DEFAULT_MIN_TAG_CONFIDENCE: f32 = 0.35;

/// TOML structure for tag category configuration.
#[derive(Debug, Deserialize)]
struct TagCategoriesConfig {
    categories: HashMap<String, CategoryKeywords>,
}

#[derive(Debug, Deserialize)]
struct CategoryKeywords {
    keywords: Vec<String>,
}

/// Lazy-loaded category keywords from embedded TOML file.
static CATEGORY_KEYWORDS: Lazy<HashMap<TagCategory, Vec<String>>> = Lazy::new(|| {
    const EMBEDDED_CONFIG: &str = include_str!("../../../.vendor/models/tag_categories.toml");

    // Parse TOML configuration
    let config: TagCategoriesConfig = toml::from_str(EMBEDDED_CONFIG)
        .expect("Failed to parse tag_categories.toml");

    // Convert string keys to TagCategory enum
    let mut category_map = HashMap::new();

    for (key, value) in config.categories {
        let category = match key.as_str() {
            "animal" => TagCategory::Animal,
            "vehicle" => TagCategory::Vehicle,
            "food" => TagCategory::Food,
            "nature" => TagCategory::Nature,
            "person" => TagCategory::Person,
            "structure" => TagCategory::Structure,
            "device" => TagCategory::Device,
            "furniture" => TagCategory::Furniture,
            "sport" => TagCategory::Sport,
            "clothing" => TagCategory::Clothing,
            "music" => TagCategory::Music,
            _ => continue, // Skip unknown categories
        };
        category_map.insert(category, value.keywords);
    }

    category_map
});

/// Configuration for the tagging classifier.
#[derive(Clone, Debug)]
pub struct TaggingConfig {
    /// Input image width
    pub input_width: u32,
    /// Input image height
    pub input_height: u32,
    /// Whether to apply ImageNet normalization
    pub normalize: bool,
    /// Input tensor layout ("NCHW" or "NHWC")
    pub layout: String,
    /// Normalization mean to apply when `normalize` is true.
    pub normalization_mean: [f32; 3],
    /// Normalization std to apply when `normalize` is true.
    pub normalization_std: [f32; 3],
    /// Whether to include batch dimension (rank-4 vs rank-3)
    pub batch_dim: bool,
    /// Whether this is a multi-label classifier (sigmoid outputs).
    /// When true, softmax is skipped and raw sigmoid probabilities are used.
    pub multi_label: bool,
    /// Minimum confidence threshold for filtering tags (0.0-1.0).
    /// Tags below this threshold will be filtered out.
    pub min_confidence: f32,
}

impl Default for TaggingConfig {
    fn default() -> Self {
        Self {
            input_width: 224,
            input_height: 224,
            normalize: true,
            layout: "NCHW".to_string(),
            normalization_mean: IMAGE_NET_MEAN,
            normalization_std: IMAGE_NET_STD,
            batch_dim: true,
            multi_label: false,
            min_confidence: DEFAULT_MIN_TAG_CONFIDENCE,
        }
    }
}

impl TaggingConfig {
    /// Create config from model input spec.
    pub fn from_specs(input: &ModelInputSpec) -> Self {
        Self {
            input_width: input.width,
            input_height: input.height,
            normalize: input.normalize,
            layout: input.layout.clone(),
            normalization_mean: input.mean.unwrap_or(IMAGE_NET_MEAN),
            normalization_std: input.std.unwrap_or(IMAGE_NET_STD),
            batch_dim: input.batch_dim,
            multi_label: false,
            min_confidence: DEFAULT_MIN_TAG_CONFIDENCE,
        }
    }

    /// Create config from model input spec with multi-label flag.
    pub fn from_specs_with_output(input: &ModelInputSpec, multi_label: bool) -> Self {
        Self {
            input_width: input.width,
            input_height: input.height,
            normalize: input.normalize,
            layout: input.layout.clone(),
            normalization_mean: input.mean.unwrap_or(IMAGE_NET_MEAN),
            normalization_std: input.std.unwrap_or(IMAGE_NET_STD),
            batch_dim: input.batch_dim,
            multi_label,
            min_confidence: DEFAULT_MIN_TAG_CONFIDENCE,
        }
    }
}

/// Image tagging classifier using ImageNet models.
pub struct TaggingClassifier {
    session: Session,
    config: TaggingConfig,
    labels: Vec<String>,
}

impl TaggingClassifier {
    /// Load the tagging classifier from an ONNX model file with default config.
    pub fn new(model_path: &Path) -> Result<Self, ClassifierError> {
        Self::with_config_and_labels(
            model_path,
            TaggingConfig::default(),
            Self::default_labels(),
        )
    }

    /// Load the tagging classifier with custom configuration.
    pub fn with_config(model_path: &Path, config: TaggingConfig) -> Result<Self, ClassifierError> {
        Self::with_config_and_labels(model_path, config, Self::default_labels())
    }

    /// Load the tagging classifier with custom configuration and label set.
    pub fn with_config_and_labels(
        model_path: &Path,
        config: TaggingConfig,
        labels: Vec<String>,
    ) -> Result<Self, ClassifierError> {
        let session = load_session(model_path)?;
        Ok(Self {
            session,
            config,
            labels,
        })
    }

    /// Classify an image and return the top tags.
    pub fn classify(
        &mut self,
        image_path: &Path,
        max_tags: usize,
    ) -> Result<Vec<ImageTag>, ClassifierError> {
        let input_size = (self.config.input_width as i32, self.config.input_height as i32);

        // Preprocess with model-specific settings including layout
        let input = preprocess_image_with_layout(
            image_path,
            input_size,
            self.config.normalize,
            &self.config.layout,
            self.config.normalization_mean,
            self.config.normalization_std,
            self.config.batch_dim,
        )?;

        // Get input name from model
        let input_name = self
            .session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_string());

        // Create tensor from ndarray
        let input_tensor =
            ort::value::Tensor::from_array(input).map_err(ClassifierError::Ort)?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![input_name => input_tensor])
            .map_err(ClassifierError::Ort)?;

        // Extract output tensor
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| ClassifierError::Processing("no output tensor found".into()))?;

        // Extract as tuple (shape, data slice)
        let (shape, logits_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(ClassifierError::Ort)?;

        // Flatten to 1D if needed (some models output [batch_size, num_classes])
        let logits: Vec<f32> = if shape.len() > 1 && shape[0] == 1 {
            // Output is [1, num_classes], extract just the class scores
            logits_slice.to_vec()
        } else if shape.len() > 1 {
            // Unexpected shape, just use as is
            logits_slice.to_vec()
        } else {
            logits_slice.to_vec()
        };
        
        // For multi-label models (e.g., WD taggers), each output is an independent probability.
        // Do NOT apply softmax as it would incorrectly normalize across all 10k+ tags.
        // However, some ONNX models output raw logits (pre-sigmoid) while others include
        // sigmoid in the graph. We detect this by checking if values exceed [0, 1] range.
        let probabilities = if self.config.multi_label {
            // Check if outputs appear to be raw logits (outside [0, 1] range)
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = logits.iter().cloned().fold(f32::INFINITY, f32::min);

            if max_val > 1.0 || min_val < 0.0 {
                // Raw logits detected - apply sigmoid to convert to probabilities
                logits.iter().map(|&x| sigmoid(x)).collect()
            } else {
                // Already probabilities - use as-is
                logits.clone()
            }
        } else {
            // Single-label: check if already probabilities or apply softmax
            let logits_sum: f32 = logits.iter().sum();
            let is_already_probabilities = (logits_sum - 1.0).abs() < 0.01;
            if is_already_probabilities {
                logits.clone()
            } else {
                softmax(&logits)
            }
        };

        // Get top-k predictions
        let mut indexed: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let tags: Vec<ImageTag> = indexed
            .into_iter()
            .take(max_tags)
            .filter(|(_, score)| *score >= self.config.min_confidence) // Filter by confidence threshold
            .filter_map(|(idx, score)| {
                self.labels.get(idx).map(|label| {
                    let (category, clean_label) = categorize_label(label);
                    ImageTag {
                        name: clean_label.to_lowercase().replace(' ', "-"),
                        label: clean_label.to_string(),
                        confidence: score,
                        category,
                    }
                })
            })
            .collect();

        Ok(tags)
    }

    /// Load default ImageNet 1000 labels from embedded text file.
    pub(super) fn default_labels() -> Vec<String> {
        const EMBEDDED_LABELS: &str = include_str!("../../../.vendor/models/imagenet_labels.txt");
        EMBEDDED_LABELS.lines().map(String::from).collect()
    }
}

/// A generated image tag with metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageTag {
    /// Normalized tag name (lowercase, hyphenated)
    pub name: String,
    /// Human-readable label
    pub label: String,
    /// Model confidence (0.0-1.0)
    pub confidence: f32,
    /// Tag category
    pub category: TagCategory,
}

/// Category for organizing tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TagCategory {
    /// Animals
    Animal,
    /// Vehicles and transportation
    Vehicle,
    /// Food and drinks
    Food,
    /// Nature and landscapes
    Nature,
    /// People and body parts
    Person,
    /// Buildings and structures
    Structure,
    /// Electronics and devices
    Device,
    /// Furniture and household items
    Furniture,
    /// Sports and activities
    Sport,
    /// Clothing and accessories
    Clothing,
    /// Musical instruments
    Music,
    /// General objects
    Object,
}

impl std::fmt::Display for TagCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Animal => write!(f, "Animal"),
            Self::Vehicle => write!(f, "Vehicle"),
            Self::Food => write!(f, "Food"),
            Self::Nature => write!(f, "Nature"),
            Self::Person => write!(f, "Person"),
            Self::Structure => write!(f, "Structure"),
            Self::Device => write!(f, "Device"),
            Self::Furniture => write!(f, "Furniture"),
            Self::Sport => write!(f, "Sport"),
            Self::Clothing => write!(f, "Clothing"),
            Self::Music => write!(f, "Music"),
            Self::Object => write!(f, "Object"),
        }
    }
}

/// Categorize a label and clean it up.
fn categorize_label(label: &str) -> (TagCategory, &str) {
    // Extract the primary term (before comma if present)
    let primary = label.split(',').next().unwrap_or(label).trim();

    // Simple keyword-based categorization using lazy-loaded keywords
    let lower = label.to_lowercase();

    // Check each category's keywords
    for (category, keywords) in CATEGORY_KEYWORDS.iter() {
        if keywords.iter().any(|k| lower.contains(k)) {
            return (*category, primary);
        }
    }

    // Default to Object if no match
    (TagCategory::Object, primary)
}


/// Ensemble classifier that runs multiple tagging models and merges results.
pub struct EnsembleTaggingClassifier {
    classifiers: Vec<TaggingClassifier>,
}

impl EnsembleTaggingClassifier {
    /// Create a new ensemble classifier from multiple model paths.
    pub fn new(model_paths: &[impl AsRef<Path>]) -> Result<Self, ClassifierError> {
        let mut classifiers = Vec::new();
        for path in model_paths {
            classifiers.push(TaggingClassifier::new(path.as_ref())?);
        }
        Ok(Self { classifiers })
    }

    /// Create ensemble with custom configurations for each model.
    pub fn with_configs(
        configs: Vec<(impl AsRef<Path>, TaggingConfig, Vec<String>)>,
    ) -> Result<Self, ClassifierError> {
        let mut classifiers = Vec::new();
        for (path, config, labels) in configs {
            classifiers.push(TaggingClassifier::with_config_and_labels(
                path.as_ref(),
                config,
                labels,
            )?);
        }
        Ok(Self { classifiers })
    }

    /// Classify an image using all models in the ensemble and merge tags.
    ///
    /// Merging strategy:
    /// - Collect all tags from all models
    /// - Group by tag name (normalized)
    /// - Average confidence scores for duplicate tags
    /// - Return top N tags by averaged confidence
    pub fn classify(
        &mut self,
        image_path: &Path,
        max_tags: usize,
    ) -> Result<Vec<ImageTag>, ClassifierError> {
        if self.classifiers.is_empty() {
            return Err(ClassifierError::Processing(
                "ensemble has no classifiers configured".to_string(),
            ));
        }

        // Run all classifiers and collect their tags
        let mut all_tags_by_model: Vec<Vec<ImageTag>> = Vec::new();
        for classifier in &mut self.classifiers {
            // Each classifier returns up to max_tags * 2 to give more variety
            let tags = classifier.classify(image_path, max_tags * 2)?;
            all_tags_by_model.push(tags);
        }

        // Get min_confidence from first classifier (all should use same threshold)
        let min_confidence = self.classifiers.first()
            .map(|c| c.config.min_confidence)
            .unwrap_or(DEFAULT_MIN_TAG_CONFIDENCE);

        // Merge tags by name (aggregate confidence scores)
        let merged_tags = super::ensemble::merge_tags(all_tags_by_model, min_confidence);

        // Sort by confidence and take top max_tags
        let mut sorted_tags = merged_tags;
        sorted_tags.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        sorted_tags.truncate(max_tags);

        Ok(sorted_tags)
    }
}
