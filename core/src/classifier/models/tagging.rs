//! Phase 1 tagging model definitions.
//!
//! This module contains the 3 Phase 1 image tagging models:
//! - smilingwolf-wd-v1-4-convnextv2-tagger-v2 (9,999 Danbooru tags)
//! - fancyfeast-joytag (3,472 Danbooru tags)
//! - cella110n-cl-tagger (42,163 tags)
//!
//! Note: Moderation models can also be used in tagging pipelines.
//! See moderation.rs for dual-purpose models like vladmandic-nudenet and deepghs-nudenet-onnx.

use crate::classifier::config::{ModelConfig, ModelInputSpec, ModelOutputSpec, ModelType};
use std::path::PathBuf;

/// Get all Phase 1 tagging models.
pub fn tagging_models() -> Vec<(&'static str, ModelConfig)> {
    vec![
        smilingwolf_wd_v1_4_convnextv2_tagger_v2(),
        fancyfeast_joytag(),
        cella110n_cl_tagger(),
    ]
}

/// SmilingWolf WD v1.4 ConvNeXtV2 Tagger V2 (9,999 Danbooru tags)
fn smilingwolf_wd_v1_4_convnextv2_tagger_v2() -> (&'static str, ModelConfig) {
    (
        "smilingwolf-wd-v1-4-convnextv2-tagger-v2",
        ModelConfig {
            name: "WD v1.4 ConvNeXtV2 Tagger V2".to_string(),
            model_type: ModelType::Tagging,
            path: PathBuf::from("smilingwolf-wd-v1-4-convnextv2-tagger-v2.onnx"),
            input: ModelInputSpec {
                width: 448,
                height: 448,
                normalize: false, // No normalization for WD taggers
                layout: "NHWC".to_string(),
                mean: None,
                std: None,
                batch_dim: true,
            },
            output: ModelOutputSpec {
                num_classes: 9999,
                labels: Vec::new(),
                labels_file: Some(PathBuf::from("selected_tags.csv")),
                format: None,
                multi_label: true, // Multi-label sigmoid outputs
            },
            description: "ConvNeXtV2-based Danbooru tagger with 9,999 tags".to_string(),
            enabled: true,
        },
    )
}

/// FancyFeast JoyTag (3,472 Danbooru tags)
fn fancyfeast_joytag() -> (&'static str, ModelConfig) {
    (
        "fancyfeast-joytag",
        ModelConfig {
            name: "FancyFeast JoyTag".to_string(),
            model_type: ModelType::Tagging,
            path: PathBuf::from("fancyfeast-joytag.onnx"),
            input: ModelInputSpec {
                width: 448,
                height: 448,
                normalize: true,
                layout: "NCHW".to_string(),
                mean: Some([0.48145466, 0.4578275, 0.40821073]),
                std: Some([0.26862954, 0.261_302_6, 0.275_777_1]),
                batch_dim: true,
            },
            output: ModelOutputSpec {
                num_classes: 3472,
                labels: Vec::new(),
                labels_file: Some(PathBuf::from("top_tags.txt")),
                format: None,
                multi_label: true,
            },
            description: "ViT-B/16 Danbooru tagger with 3,472 tags".to_string(),
            enabled: true,
        },
    )
}

/// Cella110n CL Tagger (42,163 tags across 6 categories)
fn cella110n_cl_tagger() -> (&'static str, ModelConfig) {
    (
        "cella110n-cl-tagger",
        ModelConfig {
            name: "Cella110n CL Tagger".to_string(),
            model_type: ModelType::Tagging,
            path: PathBuf::from("cl_tagger_1_02_model.onnx"),
            input: ModelInputSpec {
                width: 448,
                height: 448,
                normalize: false, // WD taggers don't use normalization
                layout: "NHWC".to_string(),
                mean: None,
                std: None,
                batch_dim: true,
            },
            output: ModelOutputSpec {
                num_classes: 42163,
                labels: Vec::new(),
                labels_file: Some(PathBuf::from("cl_tagger_1_02_tag_mapping.json")),
                format: None,
                multi_label: true,
            },
            description: "EVA02-large based tagger with 42,163 tags across 6 categories"
                .to_string(),
            enabled: true,
        },
    )
}
