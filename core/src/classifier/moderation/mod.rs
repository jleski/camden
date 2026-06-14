//! NSFW content moderation using configurable models.
//!
//! Supports multiple model formats:
//! - GantMan 5-class: drawings, hentai, neutral, porn, sexy (299x299)
//! - Falconsai 2-class: normal, nsfw (224x224)
//! - AdamCodd 2-class: nsfw, sfw (384x384)

pub mod interpreter;

use self::interpreter::{score_to_tier, validate_probabilities};
use super::config::ModelInputSpec;
use super::runtime::{
    load_session, preprocess_image_with_layout, softmax, ClassifierError, IMAGE_NET_MEAN,
    IMAGE_NET_STD,
};
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Model format enumeration for different NSFW classifiers.
#[derive(Clone, Debug, PartialEq)]
pub enum ModerationModelFormat {
    /// GantMan/NSFWJS 5-class model (drawings, hentai, neutral, porn, sexy)
    /// Both models use the same class order
    GantMan5Class,
    /// 2-class model with normal/nsfw output order
    TwoClassNormalNsfw,
    /// 2-class model with nsfw/sfw output order
    TwoClassNsfwSfw,
    /// 3-class NSFW model (Safe, NSFW Mild, NSFW Explicit)
    /// Used by n4xtan-nsfw-classification (CLIP ViT-based)
    ThreeClassNsfw,
    /// 4-tier severity model (Neutral, Low, Medium, High)
    /// Used by spiele-nsfw-image-detector (EVA02-based)
    FourTierNsfw,
    /// Generic model - use raw scores
    Generic {
        num_classes: usize,
        labels: Vec<String>,
    },
}

impl ModerationModelFormat {
    /// Detect format from explicit hint or fall back to label-based detection.
    ///
    /// # Arguments
    /// * `format_hint` - Optional explicit format string (e.g., "gantman", "nsfw_3class", "nsfw_4tier")
    /// * `labels` - Class labels for automatic format detection
    pub fn from_config(format_hint: Option<&str>, labels: &[String]) -> Self {
        // Use explicit format hint if provided
        if let Some(hint) = format_hint {
            return match hint.to_lowercase().as_str() {
                "gantman" | "gantman5" | "nsfwjs" | "nsfwjs5" => Self::GantMan5Class,
                "falconsai" | "normal_nsfw" => Self::TwoClassNormalNsfw,
                "adamcodd" | "nsfw_sfw" => Self::TwoClassNsfwSfw,
                "nsfw_3class" | "3class" | "n4xtan" => Self::ThreeClassNsfw,
                "nsfw_4tier" | "4tier" | "spiele" => Self::FourTierNsfw,
                _ => Self::from_labels(labels),
            };
        }

        Self::from_labels(labels)
    }

    /// Detect format from label configuration only.
    pub fn from_labels(labels: &[String]) -> Self {
        match labels.len() {
            3 if labels.iter().any(|l| l.to_lowercase().contains("explicit")) => {
                Self::ThreeClassNsfw
            }
            4 if labels.iter().any(|l| l.to_lowercase() == "neutral")
                && labels.iter().any(|l| l.to_lowercase() == "low") =>
            {
                Self::FourTierNsfw
            }
            5 if labels.iter().any(|l| l == "hentai") => Self::GantMan5Class,
            2 if labels.first().map(|s| s.as_str()) == Some("normal") => Self::TwoClassNormalNsfw,
            2 if labels.first().map(|s| s.as_str()) == Some("nsfw") => Self::TwoClassNsfwSfw,
            n => Self::Generic {
                num_classes: n,
                labels: labels.to_vec(),
            },
        }
    }
}

/// Configuration for the moderation classifier.
#[derive(Clone, Debug)]
pub struct ModerationConfig {
    /// Input image width
    pub input_width: u32,
    /// Input image height
    pub input_height: u32,
    /// Whether to apply ImageNet normalization
    pub normalize: bool,
    /// Input tensor layout (NCHW or NHWC)
    pub layout: String,
    /// Normalization mean to apply when `normalize` is true.
    pub normalization_mean: [f32; 3],
    /// Normalization std to apply when `normalize` is true.
    pub normalization_std: [f32; 3],
    /// Whether to include batch dimension (rank-4 vs rank-3)
    pub batch_dim: bool,
    /// Model format for output interpretation
    pub format: ModerationModelFormat,
}

impl Default for ModerationConfig {
    fn default() -> Self {
        Self {
            input_width: 299,
            input_height: 299,
            normalize: false,
            layout: "NHWC".to_string(),
            normalization_mean: IMAGE_NET_MEAN,
            normalization_std: IMAGE_NET_STD,
            batch_dim: true,
            format: ModerationModelFormat::GantMan5Class,
        }
    }
}

impl ModerationConfig {
    /// Create config from model input/output specs.
    pub fn from_specs(
        input: &ModelInputSpec,
        labels: &[String],
        format_hint: Option<&str>,
    ) -> Self {
        Self {
            input_width: input.width,
            input_height: input.height,
            normalize: input.normalize,
            layout: input.layout.clone(),
            normalization_mean: input.mean.unwrap_or(IMAGE_NET_MEAN),
            normalization_std: input.std.unwrap_or(IMAGE_NET_STD),
            batch_dim: input.batch_dim,
            format: ModerationModelFormat::from_config(format_hint, labels),
        }
    }
}

/// Content moderation classifier using NSFW detection.
pub struct NsfwClassifier {
    session: Session,
    config: ModerationConfig,
}

impl NsfwClassifier {
    /// Load the NSFW classifier from an ONNX model file with default (GantMan) config.
    pub fn new(model_path: &Path) -> Result<Self, ClassifierError> {
        Self::with_config(model_path, ModerationConfig::default())
    }

    /// Load the NSFW classifier with custom configuration.
    pub fn with_config(
        model_path: &Path,
        config: ModerationConfig,
    ) -> Result<Self, ClassifierError> {
        let session = load_session(model_path)?;
        Ok(Self { session, config })
    }

    /// Classify an image and return moderation flags.
    pub fn classify(&mut self, image_path: &Path) -> Result<ModerationFlags, ClassifierError> {
        let input_size = (
            self.config.input_width as i32,
            self.config.input_height as i32,
        );

        // Preprocess with model-specific settings (layout, normalization)
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
            .unwrap_or_else(|| "pixel_values".to_string());

        // Create tensor from ndarray
        let input_tensor = ort::value::Tensor::from_array(input).map_err(ClassifierError::Ort)?;

        // Run inference and extract scores before dropping outputs
        let probabilities = {
            let outputs = self
                .session
                .run(ort::inputs![input_name => input_tensor])
                .map_err(ClassifierError::Ort)?;

            // Extract output tensor - get first output
            let output = outputs
                .values()
                .next()
                .ok_or_else(|| ClassifierError::Processing("no output tensor found".into()))?;

            // Extract as tuple (shape, data slice)
            let (_shape, scores_slice) = output
                .try_extract_tensor::<f32>()
                .map_err(ClassifierError::Ort)?;

            let scores: Vec<f32> = scores_slice.to_vec();

            // Apply softmax if needed (check if already probabilities)
            if scores.iter().sum::<f32>() > 0.99
                && scores.iter().sum::<f32>() < 1.01
                && scores.iter().all(|&x| x >= 0.0)
            {
                scores
            } else {
                softmax(&scores)
            }
        };

        // Convert to ModerationFlags based on model format
        self.interpret_output(&probabilities)
    }

    /// Interpret model output based on the configured format.
    fn interpret_output(&self, probabilities: &[f32]) -> Result<ModerationFlags, ClassifierError> {
        match &self.config.format {
            ModerationModelFormat::GantMan5Class => {
                validate_probabilities(probabilities, 5, "GantMan5Class")?;

                let categories = ModerationCategories {
                    drawings: probabilities[0],
                    hentai: probabilities[1],
                    neutral: probabilities[2],
                    porn: probabilities[3],
                    sexy: probabilities[4],
                };

                let tier = categories.determine_tier();
                let safety_score = categories.safety_score();

                Ok(ModerationFlags {
                    tier,
                    safety_score,
                    categories,
                })
            }

            ModerationModelFormat::TwoClassNormalNsfw => {
                // Output: [normal, nsfw]
                validate_probabilities(probabilities, 2, "TwoClassNormalNsfw")?;

                let normal_score = probabilities[0];
                let nsfw_score = probabilities[1];

                let tier = score_to_tier(nsfw_score);
                let categories = ModerationCategories {
                    drawings: 0.0,
                    hentai: 0.0,
                    neutral: normal_score,
                    porn: nsfw_score,
                    sexy: 0.0,
                };

                Ok(ModerationFlags {
                    tier,
                    safety_score: nsfw_score,
                    categories,
                })
            }

            ModerationModelFormat::ThreeClassNsfw => {
                // Output: [Safe, NSFW Mild, NSFW Explicit]
                // Used by n4xtan-nsfw-classification
                validate_probabilities(probabilities, 3, "ThreeClassNsfw")?;

                let safe_score = probabilities[0];
                let mild_score = probabilities[1];
                let explicit_score = probabilities[2];

                // Use highest NSFW score for tier determination
                let nsfw_score = explicit_score.max(mild_score);
                let (tier, safety_score) = if explicit_score > 0.5 {
                    (ModerationTier::Restricted, explicit_score)
                } else if mild_score > 0.5 {
                    (ModerationTier::Sensitive, mild_score)
                } else {
                    (score_to_tier(nsfw_score), nsfw_score)
                };

                let categories = ModerationCategories {
                    drawings: 0.0,
                    hentai: 0.0,
                    neutral: safe_score,
                    porn: explicit_score,
                    sexy: mild_score,
                };

                Ok(ModerationFlags {
                    tier,
                    safety_score,
                    categories,
                })
            }

            ModerationModelFormat::FourTierNsfw => {
                // Output: [Neutral, Low, Medium, High]
                // Used by spiele-nsfw-image-detector
                validate_probabilities(probabilities, 4, "FourTierNsfw")?;

                let neutral_score = probabilities[0];
                let low_score = probabilities[1];
                let medium_score = probabilities[2];
                let high_score = probabilities[3];

                // Map 4 tiers to our moderation system (custom thresholds for this model)
                let (tier, safety_score) = if high_score > 0.5 {
                    (ModerationTier::Restricted, high_score)
                } else if medium_score > 0.4 {
                    (ModerationTier::Mature, medium_score)
                } else if low_score > 0.3 {
                    (ModerationTier::Sensitive, low_score)
                } else {
                    (
                        ModerationTier::Safe,
                        high_score.max(medium_score).max(low_score),
                    )
                };

                let categories = ModerationCategories {
                    drawings: 0.0,
                    hentai: medium_score,
                    neutral: neutral_score,
                    porn: high_score,
                    sexy: low_score,
                };

                Ok(ModerationFlags {
                    tier,
                    safety_score,
                    categories,
                })
            }

            ModerationModelFormat::TwoClassNsfwSfw => {
                // Output: [nsfw, sfw]
                validate_probabilities(probabilities, 2, "TwoClassNsfwSfw")?;

                let nsfw_score = probabilities[0];
                let sfw_score = probabilities[1];

                let tier = score_to_tier(nsfw_score);
                let categories = ModerationCategories {
                    drawings: 0.0,
                    hentai: 0.0,
                    neutral: sfw_score,
                    porn: nsfw_score,
                    sexy: 0.0,
                };

                Ok(ModerationFlags {
                    tier,
                    safety_score: nsfw_score,
                    categories,
                })
            }

            ModerationModelFormat::Generic {
                num_classes,
                labels,
            } => {
                validate_probabilities(probabilities, *num_classes, "Generic")?;

                // Find the highest scoring class
                let (max_idx, max_score) = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap_or((0, &0.0));

                let label = labels.get(max_idx).map(|s| s.as_str()).unwrap_or("unknown");

                // Infer tier from label name
                let tier = match label.to_lowercase().as_str() {
                    s if s.contains("nsfw") || s.contains("porn") || s.contains("explicit") => {
                        ModerationTier::Restricted
                    }
                    s if s.contains("hentai") || s.contains("mature") => ModerationTier::Mature,
                    s if s.contains("sexy")
                        || s.contains("suggestive")
                        || s.contains("sensual") =>
                    {
                        ModerationTier::Sensitive
                    }
                    _ => ModerationTier::Safe,
                };

                let categories = ModerationCategories {
                    drawings: 0.0,
                    hentai: 0.0,
                    neutral: if tier == ModerationTier::Safe {
                        *max_score
                    } else {
                        0.0
                    },
                    porn: if tier == ModerationTier::Restricted {
                        *max_score
                    } else {
                        0.0
                    },
                    sexy: if tier == ModerationTier::Sensitive {
                        *max_score
                    } else {
                        0.0
                    },
                };

                Ok(ModerationFlags {
                    tier,
                    safety_score: if tier == ModerationTier::Safe {
                        0.0
                    } else {
                        *max_score
                    },
                    categories,
                })
            }
        }
    }
}

/// Moderation tier indicating content safety level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModerationTier {
    /// Safe content (neutral, drawings)
    Safe,
    /// Mildly suggestive content (sexy)
    Sensitive,
    /// Adult/explicit illustrated content (hentai)
    Mature,
    /// Explicit photographic content (porn)
    Restricted,
}

impl ModerationTier {
    /// Returns a numeric level for comparison (0 = safest, 3 = most restricted).
    pub fn level(&self) -> u8 {
        match self {
            Self::Safe => 0,
            Self::Sensitive => 1,
            Self::Mature => 2,
            Self::Restricted => 3,
        }
    }
}

impl std::fmt::Display for ModerationTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safe => write!(f, "Safe"),
            Self::Sensitive => write!(f, "Sensitive"),
            Self::Mature => write!(f, "Mature"),
            Self::Restricted => write!(f, "Restricted"),
        }
    }
}

/// Individual category scores from the NSFW model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModerationCategories {
    /// Safe illustrations, anime
    pub drawings: f32,
    /// Explicit illustrated content
    pub hentai: f32,
    /// Safe, neutral images
    pub neutral: f32,
    /// Explicit photographic content
    pub porn: f32,
    /// Suggestive but not explicit
    pub sexy: f32,
}

impl ModerationCategories {
    /// Get the dominant (highest scoring) class.
    pub fn dominant_class(&self) -> &'static str {
        let scores = [
            (self.drawings, "drawings"),
            (self.hentai, "hentai"),
            (self.neutral, "neutral"),
            (self.porn, "porn"),
            (self.sexy, "sexy"),
        ];
        scores
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, name)| name)
            .unwrap_or("neutral")
    }

    /// Calculate the moderation tier based on category scores.
    pub fn determine_tier(&self) -> ModerationTier {
        let dominant = self.dominant_class();
        let dominant_score = match dominant {
            "drawings" => self.drawings,
            "hentai" => self.hentai,
            "neutral" => self.neutral,
            "porn" => self.porn,
            "sexy" => self.sexy,
            _ => 0.0,
        };

        // Require minimum confidence for non-safe classifications
        const MIN_CONFIDENCE: f32 = 0.3;

        if dominant_score < MIN_CONFIDENCE {
            // Low confidence, default to safe
            return ModerationTier::Safe;
        }

        match dominant {
            "neutral" | "drawings" => ModerationTier::Safe,
            "sexy" => ModerationTier::Sensitive,
            "hentai" => ModerationTier::Mature,
            "porn" => ModerationTier::Restricted,
            _ => ModerationTier::Safe,
        }
    }

    /// Calculate overall safety score (0.0 = safe, 1.0 = unsafe).
    pub fn safety_score(&self) -> f32 {
        // Weighted combination of unsafe categories
        let unsafe_score = self.porn * 1.0 + self.hentai * 0.9 + self.sexy * 0.4;
        unsafe_score.min(1.0)
    }
}

/// Complete moderation flags for an image.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModerationFlags {
    /// Assigned moderation tier
    pub tier: ModerationTier,
    /// Overall safety score (0.0 = safe, 1.0 = unsafe)
    pub safety_score: f32,
    /// Individual category scores
    pub categories: ModerationCategories,
}

/// Ensemble classifier that runs multiple moderation models and aggregates results.
pub struct EnsembleModerationClassifier {
    classifiers: Vec<NsfwClassifier>,
    aggregation: AggregationStrategy,
}

/// Strategy for aggregating results from multiple models.
#[derive(Clone, Copy, Debug, Default)]
pub enum AggregationStrategy {
    /// Average probability scores across all models (default).
    #[default]
    Average,
    /// Use the maximum (most conservative) tier across all models
    MaxTier,
    /// Use weighted average with configurable weights per model
    Weighted,
}

impl EnsembleModerationClassifier {
    /// Create a new ensemble classifier from multiple model paths.
    pub fn new(model_paths: &[impl AsRef<Path>]) -> Result<Self, ClassifierError> {
        let mut classifiers = Vec::new();
        for path in model_paths {
            classifiers.push(NsfwClassifier::new(path.as_ref())?);
        }
        Ok(Self {
            classifiers,
            aggregation: AggregationStrategy::default(),
        })
    }

    /// Create ensemble with custom configurations for each model.
    pub fn with_configs(
        configs: Vec<(impl AsRef<Path>, ModerationConfig)>,
    ) -> Result<Self, ClassifierError> {
        let mut classifiers = Vec::new();
        for (path, config) in configs {
            classifiers.push(NsfwClassifier::with_config(path.as_ref(), config)?);
        }
        Ok(Self {
            classifiers,
            aggregation: AggregationStrategy::default(),
        })
    }

    /// Set the aggregation strategy.
    pub fn with_strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation = strategy;
        self
    }

    /// Classify an image using all models in the ensemble.
    pub fn classify(&mut self, image_path: &Path) -> Result<ModerationFlags, ClassifierError> {
        if self.classifiers.is_empty() {
            return Err(ClassifierError::Processing(
                "ensemble has no classifiers configured".to_string(),
            ));
        }

        // Run all classifiers
        let mut results = Vec::new();
        for classifier in self.classifiers.iter_mut() {
            results.push(classifier.classify(image_path)?);
        }

        // Aggregate results based on strategy
        match self.aggregation {
            AggregationStrategy::Average => self.aggregate_average(&results),
            AggregationStrategy::MaxTier => self.aggregate_max_tier(&results),
            AggregationStrategy::Weighted => self.aggregate_average(&results), // TODO: Add weights
        }
    }

    /// Aggregate results by averaging category scores.
    fn aggregate_average(
        &self,
        results: &[ModerationFlags],
    ) -> Result<ModerationFlags, ClassifierError> {
        let n = results.len() as f32;

        // Average each category score
        let avg_categories = ModerationCategories {
            drawings: results.iter().map(|r| r.categories.drawings).sum::<f32>() / n,
            hentai: results.iter().map(|r| r.categories.hentai).sum::<f32>() / n,
            neutral: results.iter().map(|r| r.categories.neutral).sum::<f32>() / n,
            porn: results.iter().map(|r| r.categories.porn).sum::<f32>() / n,
            sexy: results.iter().map(|r| r.categories.sexy).sum::<f32>() / n,
        };

        let tier = avg_categories.determine_tier();
        let safety_score = avg_categories.safety_score();

        Ok(ModerationFlags {
            tier,
            safety_score,
            categories: avg_categories,
        })
    }

    /// Aggregate results by taking the maximum (most restrictive) tier.
    fn aggregate_max_tier(
        &self,
        results: &[ModerationFlags],
    ) -> Result<ModerationFlags, ClassifierError> {
        // Find the result with the highest tier level
        let max_result = results
            .iter()
            .max_by_key(|r| r.tier.level())
            .ok_or_else(|| ClassifierError::Processing("no results to aggregate".to_string()))?;

        // Also average the category scores for additional context
        let n = results.len() as f32;
        let avg_categories = ModerationCategories {
            drawings: results.iter().map(|r| r.categories.drawings).sum::<f32>() / n,
            hentai: results.iter().map(|r| r.categories.hentai).sum::<f32>() / n,
            neutral: results.iter().map(|r| r.categories.neutral).sum::<f32>() / n,
            porn: results.iter().map(|r| r.categories.porn).sum::<f32>() / n,
            sexy: results.iter().map(|r| r.categories.sexy).sum::<f32>() / n,
        };

        Ok(ModerationFlags {
            tier: max_result.tier,
            safety_score: max_result.safety_score,
            categories: avg_categories,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_from_categories() {
        let safe = ModerationCategories {
            drawings: 0.1,
            hentai: 0.0,
            neutral: 0.85,
            porn: 0.0,
            sexy: 0.05,
        };
        assert_eq!(safe.determine_tier(), ModerationTier::Safe);

        let sensitive = ModerationCategories {
            drawings: 0.1,
            hentai: 0.05,
            neutral: 0.1,
            porn: 0.05,
            sexy: 0.7,
        };
        assert_eq!(sensitive.determine_tier(), ModerationTier::Sensitive);

        let mature = ModerationCategories {
            drawings: 0.05,
            hentai: 0.8,
            neutral: 0.05,
            porn: 0.05,
            sexy: 0.05,
        };
        assert_eq!(mature.determine_tier(), ModerationTier::Mature);

        let restricted = ModerationCategories {
            drawings: 0.0,
            hentai: 0.1,
            neutral: 0.0,
            porn: 0.85,
            sexy: 0.05,
        };
        assert_eq!(restricted.determine_tier(), ModerationTier::Restricted);
    }

    #[test]
    fn test_low_confidence_defaults_to_safe() {
        // When no class has high confidence, default to Safe
        let uncertain = ModerationCategories {
            drawings: 0.2,
            hentai: 0.2,
            neutral: 0.2,
            porn: 0.2,
            sexy: 0.2,
        };
        assert_eq!(uncertain.determine_tier(), ModerationTier::Safe);
    }

    #[test]
    fn test_format_detection() {
        let gantman = ModerationModelFormat::from_labels(&[
            "drawings".into(),
            "hentai".into(),
            "neutral".into(),
            "porn".into(),
            "sexy".into(),
        ]);
        assert_eq!(gantman, ModerationModelFormat::GantMan5Class);

        let falconsai = ModerationModelFormat::from_labels(&["normal".into(), "nsfw".into()]);
        assert_eq!(falconsai, ModerationModelFormat::TwoClassNormalNsfw);

        let adamcodd = ModerationModelFormat::from_labels(&["nsfw".into(), "sfw".into()]);
        assert_eq!(adamcodd, ModerationModelFormat::TwoClassNsfwSfw);
    }

    #[test]
    fn test_format_explicit_hint() {
        let labels = vec![
            "drawings".into(),
            "hentai".into(),
            "neutral".into(),
            "porn".into(),
            "sexy".into(),
        ];

        // Both GantMan and NSFWJS hints should resolve to the same 5-class format
        let gantman = ModerationModelFormat::from_config(Some("gantman"), &labels);
        assert_eq!(gantman, ModerationModelFormat::GantMan5Class);

        let nsfwjs = ModerationModelFormat::from_config(Some("nsfwjs"), &labels);
        assert_eq!(nsfwjs, ModerationModelFormat::GantMan5Class);

        // Without hint, should detect GantMan5Class from labels
        let auto = ModerationModelFormat::from_config(None, &labels);
        assert_eq!(auto, ModerationModelFormat::GantMan5Class);
    }
}
