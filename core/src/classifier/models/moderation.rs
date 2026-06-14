//! Phase 1 moderation model definitions.
//!
//! This module contains the 5 Phase 1 NSFW classification models:
//! - n4xtan-nsfw-classification (CLIP-based)
//! - spiele-nsfw-image-detector (4-tier EVA02)
//! - vladmandic-nudenet (body part detection) ★ Dual-purpose: also useful for tagging
//! - deepghs-nudenet-onnx (YOLOv8-based detection) ★ Dual-purpose: also useful for tagging
//! - taufiqdp-mobilenetv4-nsfw (5-class MobileNetV4)
//!
//! Note: Models marked with ★ can be used in both moderation and tagging pipelines.
//! All moderation models have inline labels and can technically be used for tagging.

use crate::classifier::config::{ModelConfig, ModelInputSpec, ModelOutputSpec, ModelType};
use std::path::PathBuf;

/// Get all Phase 1 moderation models.
pub fn moderation_models() -> Vec<(&'static str, ModelConfig)> {
    vec![
        n4xtan_nsfw_classification(),
        spiele_nsfw_image_detector(),
        vladmandic_nudenet(),
        deepghs_nudenet_onnx(),
        taufiqdp_mobilenetv4_nsfw(),
    ]
}

/// n4xtan NSFW Classification (CLIP-based)
fn n4xtan_nsfw_classification() -> (&'static str, ModelConfig) {
    (
        "n4xtan-nsfw-classification",
        ModelConfig {
            name: "n4xtan NSFW Classification".to_string(),
            model_type: ModelType::Moderation,
            path: PathBuf::from("nsfw_model.onnx"),
            input: ModelInputSpec {
                width: 224,
                height: 224,
                normalize: true,
                layout: "NCHW".to_string(),
                mean: Some([0.48145466, 0.4578275, 0.40821073]),
                std: Some([0.26862954, 0.261_302_6, 0.275_777_1]),
                batch_dim: true,
            },
            output: ModelOutputSpec {
                num_classes: 2,
                labels: vec!["normal".to_string(), "nsfw".to_string()],
                labels_file: None,
                format: Some("binary".to_string()),
                multi_label: false,
            },
            description: "CLIP-based binary NSFW classifier".to_string(),
            enabled: true,
        },
    )
}

/// spiele NSFW Image Detector (4-tier: Neutral, Low, Medium, High)
fn spiele_nsfw_image_detector() -> (&'static str, ModelConfig) {
    (
        "spiele-nsfw-image-detector",
        ModelConfig {
            name: "spiele NSFW Image Detector".to_string(),
            model_type: ModelType::Moderation,
            path: PathBuf::from("spiele-nsfw-image-detector.onnx"),
            input: ModelInputSpec {
                width: 448,
                height: 448,
                normalize: true,
                layout: "NCHW".to_string(),
                mean: Some([0.48, 0.46, 0.41]),
                std: Some([0.27, 0.26, 0.28]),
                batch_dim: true,
            },
            output: ModelOutputSpec {
                num_classes: 4,
                labels: vec![
                    "Neutral".to_string(),
                    "Low".to_string(),
                    "Medium".to_string(),
                    "High".to_string(),
                ],
                labels_file: None,
                format: Some("nsfw_4tier".to_string()),
                multi_label: false,
            },
            description: "EVA02-based 4-tier NSFW severity detector (Neutral → High)".to_string(),
            enabled: true,
        },
    )
}

/// Vladmandic NudeNet (body part detection)
fn vladmandic_nudenet() -> (&'static str, ModelConfig) {
    (
        "vladmandic-nudenet",
        ModelConfig {
            name: "Vladmandic NudeNet".to_string(),
            model_type: ModelType::Moderation,
            path: PathBuf::from("nudenet.onnx"),
            input: ModelInputSpec {
                width: 320,
                height: 320,
                normalize: true,
                layout: "NCHW".to_string(),
                mean: None,
                std: None,
                batch_dim: true, // NudeNet expects rank-4 tensor [N, C, H, W]
            },
            output: ModelOutputSpec {
                num_classes: 12,    // Body part detection classes
                labels: Vec::new(), // Labels embedded in model
                labels_file: None,
                format: Some("detection".to_string()),
                multi_label: true,
            },
            description: "Body part detection for content moderation (12 classes)".to_string(),
            enabled: true,
        },
    )
}

/// DeepGHS NudeNet ONNX (YOLOv8-based detection)
fn deepghs_nudenet_onnx() -> (&'static str, ModelConfig) {
    (
        "deepghs-nudenet-onnx",
        ModelConfig {
            name: "DeepGHS NudeNet ONNX".to_string(),
            model_type: ModelType::Moderation,
            path: PathBuf::from("deepghs-320n.onnx"),
            input: ModelInputSpec {
                width: 320,
                height: 320,
                normalize: true,
                layout: "NCHW".to_string(),
                mean: None,
                std: None,
                batch_dim: false, // YOLOv8 expects rank-3 tensor [C, H, W]
            },
            output: ModelOutputSpec {
                num_classes: 12,
                labels: Vec::new(),
                labels_file: None,
                format: Some("detection".to_string()),
                multi_label: true,
            },
            description: "YOLOv8-based body part detection for moderation".to_string(),
            enabled: true,
        },
    )
}

/// TaufiqDP MobileNetV4 NSFW (5-class GantMan schema)
fn taufiqdp_mobilenetv4_nsfw() -> (&'static str, ModelConfig) {
    (
        "taufiqdp-mobilenetv4-nsfw",
        ModelConfig {
            name: "TaufiqDP MobileNetV4 NSFW".to_string(),
            model_type: ModelType::Moderation,
            path: PathBuf::from("taufiqdp-mobilenetv4-nsfw.onnx"),
            input: ModelInputSpec {
                width: 224,
                height: 224,
                normalize: true,
                layout: "NCHW".to_string(),
                mean: None, // Uses ImageNet standard
                std: None,
                batch_dim: false, // MobileNetV4 expects rank-3 tensor [C, H, W]
            },
            output: ModelOutputSpec {
                num_classes: 5,
                labels: vec![
                    "drawings".to_string(),
                    "hentai".to_string(),
                    "neutral".to_string(),
                    "porn".to_string(),
                    "sexy".to_string(),
                ],
                labels_file: None,
                format: Some("gantman".to_string()),
                multi_label: false,
            },
            description: "MobileNetV4-based NSFW classifier (GantMan 5-class schema)".to_string(),
            enabled: true,
        },
    )
}
