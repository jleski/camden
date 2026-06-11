//! Label loading strategies for tagging classifiers.
//!
//! This module provides functions for loading label files in various formats:
//! - CSV (with header parsing)
//! - Plain text (one label per line)
//! - JSON (future support)
//! - Inline from configuration

use super::config::ModelOutputSpec;
use super::runtime::ClassifierError;
use super::tagging::TaggingClassifier;
use csv::ReaderBuilder;
use std::fs;
use std::path::Path;

/// Load labels from model output specification.
///
/// This is the main entry point for loading labels. It handles:
/// 1. Inline labels from config
/// 2. External label files (CSV, TXT, JSON)
/// 3. Default labels fallback
///
/// # Arguments
///
/// * `models_dir` - Base directory for resolving relative label file paths
/// * `output` - Model output specification with labels or label file reference
///
/// # Returns
///
/// A vector of label strings
///
/// # Examples
///
/// ```ignore
/// let labels = load_labels(&models_dir, &output_spec)?;
/// ```
pub fn load_labels(
    models_dir: &Path,
    output: &ModelOutputSpec,
) -> Result<Vec<String>, ClassifierError> {
    // First check for inline labels
    if !output.labels.is_empty() {
        return Ok(output.labels.clone());
    }

    // Then check for label file
    if let Some(labels_file) = &output.labels_file {
        let label_path = if labels_file.is_absolute() {
            labels_file.clone()
        } else {
            models_dir.join(labels_file)
        };

        if !label_path.exists() {
            return Err(ClassifierError::ModelNotFound(label_path));
        }

        // Detect format by extension
        if let Some(ext) = label_path.extension().and_then(|s| s.to_str()) {
            return match ext.to_lowercase().as_str() {
                "csv" => load_labels_from_csv(&label_path),
                "txt" => load_labels_from_text(&label_path),
                "json" => load_labels_from_json(&label_path),
                _ => {
                    // Unknown extension, try as plain text
                    load_labels_from_text(&label_path)
                }
            };
        }

        // No extension, try as plain text
        return load_labels_from_text(&label_path);
    }

    // Fallback to default labels
    Ok(TaggingClassifier::default_labels())
}

/// Load labels from a CSV file.
///
/// Expects a CSV with headers where the second column contains the label names.
/// This format is commonly used by WD taggers (e.g., wd-tags.csv).
///
/// # Arguments
///
/// * `path` - Path to the CSV file
///
/// # Returns
///
/// A vector of label strings from the second column
///
/// # Format
///
/// ```csv
/// id,name,category,count
/// 0,1girl,character,12345
/// 1,solo,meta,8765
/// ```
pub fn load_labels_from_csv(path: &Path) -> Result<Vec<String>, ClassifierError> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| {
            ClassifierError::Processing(format!(
                "failed to read labels CSV {}: {}",
                path.display(),
                e
            ))
        })?;

    let mut labels = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|e| {
            ClassifierError::Processing(format!("invalid label record: {}", e))
        })?;

        // Extract label from second column (index 1)
        if let Some(name) = record.get(1) {
            let trimmed = name.trim();
            if !trimmed.is_empty() {
                labels.push(trimmed.to_string());
            }
        }
    }

    if labels.is_empty() {
        return Err(ClassifierError::Processing(format!(
            "no labels found in {}",
            path.display()
        )));
    }

    Ok(labels)
}

/// Load labels from a plain text file.
///
/// Expects one label per line. Empty lines and whitespace are trimmed.
///
/// # Arguments
///
/// * `path` - Path to the text file
///
/// # Returns
///
/// A vector of label strings
///
/// # Format
///
/// ```text
/// tench
/// goldfish
/// great white shark
/// tiger shark
/// ```
pub fn load_labels_from_text(path: &Path) -> Result<Vec<String>, ClassifierError> {
    let content = fs::read_to_string(path).map_err(|e| {
        ClassifierError::Processing(format!(
            "failed to read label file {}: {}",
            path.display(),
            e
        ))
    })?;

    let labels: Vec<String> = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(String::from)
        .collect();

    if labels.is_empty() {
        return Err(ClassifierError::Processing(format!(
            "no labels found in {}",
            path.display()
        )));
    }

    Ok(labels)
}

/// Load labels from a JSON file.
///
/// Supports multiple JSON formats:
///
/// # Arguments
///
/// * `path` - Path to the JSON file
///
/// # Returns
///
/// A vector of label strings
///
/// # Formats
///
/// Array format:
/// ```json
/// ["label1", "label2", "label3"]
/// ```
///
/// Object with labels field:
/// ```json
/// {
///   "labels": ["label1", "label2", "label3"]
/// }
/// ```
///
/// Index-mapped format (CL Tagger style):
/// ```json
/// {
///   "0": {"tag": "general", "category": "Rating"},
///   "1": {"tag": "sensitive", "category": "Rating"},
///   "4": {"tag": "1girl", "category": "General"}
/// }
/// ```
pub fn load_labels_from_json(path: &Path) -> Result<Vec<String>, ClassifierError> {
    let content = fs::read_to_string(path).map_err(|e| {
        ClassifierError::Processing(format!(
            "failed to read JSON file {}: {}",
            path.display(),
            e
        ))
    })?;

    // Try parsing as array first
    if let Ok(labels) = serde_json::from_str::<Vec<String>>(&content) {
        if !labels.is_empty() {
            return Ok(labels);
        }
    }

    // Try parsing as object with "labels" field
    #[derive(serde::Deserialize)]
    struct LabelsObject {
        labels: Vec<String>,
    }

    if let Ok(obj) = serde_json::from_str::<LabelsObject>(&content) {
        if !obj.labels.is_empty() {
            return Ok(obj.labels);
        }
    }

    // Try parsing as index-mapped object (CL Tagger format)
    // Format: {"0": {"tag": "...", "category": "..."}, "1": {...}, ...}
    #[derive(serde::Deserialize)]
    struct TagEntry {
        tag: String,
        #[allow(dead_code)]
        category: Option<String>,
    }

    if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, TagEntry>>(&content) {
        // Parse indices and sort by numeric order
        let mut entries: Vec<(usize, String)> = map
            .into_iter()
            .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v.tag)))
            .collect();

        if !entries.is_empty() {
            entries.sort_by_key(|(i, _)| *i);

            // Build label vector, filling gaps with empty strings for missing indices
            let max_idx = entries.last().map(|(i, _)| *i).unwrap_or(0);
            let mut labels = vec![String::new(); max_idx + 1];

            for (idx, tag) in entries {
                labels[idx] = tag;
            }

            // Filter out empty strings (sparse indices)
            let non_empty_count = labels.iter().filter(|s| !s.is_empty()).count();
            if non_empty_count > 0 {
                return Ok(labels);
            }
        }
    }

    Err(ClassifierError::Processing(format!(
        "invalid JSON label format in {}",
        path.display()
    )))
}
