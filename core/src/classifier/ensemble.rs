//! Generic ensemble utilities for aggregating results from multiple models.
//!
//! This module provides reusable functions for combining predictions from
//! multiple classifiers using various aggregation strategies.

use super::moderation::{ModerationCategories, ModerationFlags};
use super::tagging::ImageTag;
use std::collections::HashMap;

/// Average moderation category scores across multiple results.
///
/// This function computes the mean of each category field across all provided
/// moderation results, producing a single averaged `ModerationCategories` struct.
///
/// # Arguments
///
/// * `results` - Slice of moderation results to average
///
/// # Returns
///
/// A new `ModerationCategories` with averaged scores
///
/// # Examples
///
/// ```ignore
/// let averaged = average_moderation_categories(&results);
/// ```
#[allow(dead_code)]
pub fn average_moderation_categories(results: &[ModerationFlags]) -> ModerationCategories {
    let n = results.len() as f32;

    ModerationCategories {
        drawings: results.iter().map(|r| r.categories.drawings).sum::<f32>() / n,
        hentai: results.iter().map(|r| r.categories.hentai).sum::<f32>() / n,
        neutral: results.iter().map(|r| r.categories.neutral).sum::<f32>() / n,
        porn: results.iter().map(|r| r.categories.porn).sum::<f32>() / n,
        sexy: results.iter().map(|r| r.categories.sexy).sum::<f32>() / n,
    }
}

/// Aggregate moderation results by averaging category scores.
///
/// Computes the average of all category scores across models and determines
/// the tier based on the averaged categories.
///
/// # Arguments
///
/// * `results` - Slice of moderation results from different models
///
/// # Returns
///
/// A single `ModerationFlags` with averaged scores and determined tier
///
/// # Examples
///
/// ```ignore
/// let aggregated = aggregate_by_average(&model_results)?;
/// ```
#[allow(dead_code)]
pub fn aggregate_by_average(results: &[ModerationFlags]) -> ModerationFlags {
    let avg_categories = average_moderation_categories(results);
    let tier = avg_categories.determine_tier();
    let safety_score = avg_categories.safety_score();

    ModerationFlags {
        tier,
        safety_score,
        categories: avg_categories,
    }
}

/// Aggregate moderation results by selecting the highest tier.
///
/// Chooses the result with the most restrictive tier, but averages category
/// scores across all models for additional context.
///
/// # Arguments
///
/// * `results` - Slice of moderation results from different models
///
/// # Returns
///
/// A `ModerationFlags` with the highest tier and averaged category scores
///
/// # Examples
///
/// ```ignore
/// let aggregated = aggregate_by_max_tier(&model_results);
/// ```
#[allow(dead_code)]
pub fn aggregate_by_max_tier(results: &[ModerationFlags]) -> ModerationFlags {
    // Find the result with the highest tier level
    let max_result = results
        .iter()
        .max_by_key(|r| r.tier.level())
        .expect("results cannot be empty");

    // Average category scores for additional context
    let avg_categories = average_moderation_categories(results);

    ModerationFlags {
        tier: max_result.tier,
        safety_score: max_result.safety_score,
        categories: avg_categories,
    }
}

/// Merge tags from multiple models by averaging confidence scores with agreement weighting.
///
/// When multiple models produce the same tag (matched by name), this function
/// averages their confidence scores and weights by model agreement. Tags detected
/// by more models are boosted, while tags detected by fewer models are penalized.
///
/// # Agreement Weighting
///
/// The final confidence is calculated as:
/// ```text
/// weighted_confidence = avg_confidence * (0.5 + 0.5 * agreement_ratio)
/// ```
///
/// Where `agreement_ratio = models_detecting_tag / total_models`.
///
/// This means:
/// - A tag detected by all 3 models gets full confidence (multiplied by 1.0)
/// - A tag detected by 2 of 3 models gets ~83% of its average confidence
/// - A tag detected by 1 of 3 models gets ~67% of its average confidence
///
/// # Arguments
///
/// * `all_tags` - Vector of tag vectors from different models
/// * `min_confidence` - Minimum confidence threshold for filtering tags (0.0-1.0)
///
/// # Returns
///
/// A vector of merged tags with agreement-weighted confidence scores, sorted by confidence
///
/// # Examples
///
/// ```ignore
/// let merged = merge_tags(vec![model1_tags, model2_tags, model3_tags], 0.35);
/// ```
pub fn merge_tags(all_tags: Vec<Vec<ImageTag>>, min_confidence: f32) -> Vec<ImageTag> {
    let total_models = all_tags.len();
    if total_models == 0 {
        return Vec::new();
    }

    // Group tags by name
    let mut tag_map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut tag_prototypes: HashMap<String, ImageTag> = HashMap::new();

    for tags in all_tags {
        for tag in tags {
            tag_map
                .entry(tag.name.clone())
                .or_insert_with(Vec::new)
                .push(tag.confidence);

            // Keep first occurrence as prototype (for label, category)
            tag_prototypes.entry(tag.name.clone()).or_insert(tag);
        }
    }

    // Average confidence scores with agreement weighting
    let mut merged: Vec<ImageTag> = tag_map
        .into_iter()
        .filter_map(|(name, confidences)| {
            let models_detecting = confidences.len();
            let avg_confidence = confidences.iter().sum::<f32>() / models_detecting as f32;

            // Weight by model agreement: tags detected by more models get boosted
            // Formula: 0.5 base + 0.5 * agreement_ratio
            // This gives a range from 0.5 (1 model) to 1.0 (all models)
            let agreement_ratio = models_detecting as f32 / total_models as f32;
            let weighted_confidence = avg_confidence * (0.5 + 0.5 * agreement_ratio);

            // Only return tags that meet minimum confidence after weighting
            if weighted_confidence >= min_confidence {
                tag_prototypes.get(&name).map(|prototype| ImageTag {
                    name: prototype.name.clone(),
                    label: prototype.label.clone(),
                    confidence: weighted_confidence,
                    category: prototype.category,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by confidence descending
    merged.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    merged
}
