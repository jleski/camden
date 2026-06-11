use crate::aspect_ratio::{get_aspect_ratio_priority, AspectRatioPriority};
use crate::classifier::{self, ClassifierConfig, ImageClassifier};
use crate::detector::{
    DetectorConfig, DuplicateDetector, ImageAnalysis, ImageFeatures, ImageMetadata, MatchResult,
};
use crate::rename::ensure_guid_name;
use crate::resolution::{resolution_tier, ResolutionTier};
use crate::thumbnails::ThumbnailCache;
use indicatif::ProgressBar;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use walkdir::WalkDir;

const BUCKET_PREFIX_BITS: u32 = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadingMode {
    Parallel,
    Sequential,
}

/// Parameters that control how the scanning pipeline behaves.
#[derive(Clone, Debug)]
pub struct ScanConfig {
    pub extensions: Vec<String>,
    pub threading: ThreadingMode,
    pub thumbnail_cache_root: Option<PathBuf>,
    /// When true, renames files to GUID format before processing.
    pub rename_to_guid: bool,
    /// When true, tags images below FHD resolution thresholds.
    pub detect_low_resolution: bool,
    /// When true, runs AI classification (moderation + tagging) on each image.
    pub enable_classification: bool,
    /// When true, enables feature-based detection using ORB + RANSAC (finds crops).
    pub enable_feature_detection: bool,
    /// When true, prefers originals with standard display aspect ratios (e.g., 16:9).
    pub prefer_display_aspect_ratios: bool,
}

impl ScanConfig {
    /// Builds a new configuration from the supplied extensions and threading mode.
    pub fn new(extensions: Vec<String>, threading: ThreadingMode) -> Self {
        Self {
            extensions,
            threading,
            thumbnail_cache_root: None,
            rename_to_guid: false,
            detect_low_resolution: false,
            enable_classification: false,
            enable_feature_detection: false,
            prefer_display_aspect_ratios: false,
        }
    }

    pub fn with_thumbnail_root(mut self, root: PathBuf) -> Self {
        self.thumbnail_cache_root = Some(root);
        self
    }

    pub fn with_guid_rename(mut self, enabled: bool) -> Self {
        self.rename_to_guid = enabled;
        self
    }

    pub fn with_low_resolution_detection(mut self, enabled: bool) -> Self {
        self.detect_low_resolution = enabled;
        self
    }

    pub fn with_classification(mut self, enabled: bool) -> Self {
        self.enable_classification = enabled;
        self
    }

    pub fn with_feature_detection(mut self, enabled: bool) -> Self {
        self.enable_feature_detection = enabled;
        self
    }

    pub fn with_prefer_display_aspect_ratios(mut self, enabled: bool) -> Self {
        self.prefer_display_aspect_ratios = enabled;
        self
    }
}

/// Maximum number of tags to keep per image.
pub const MAX_TAGS_PER_IMAGE: usize = 5;

/// Whether to include confidence scores in tag strings (e.g., "boat (85%)") for debugging.
pub const DEBUG_SHOW_TAG_CONFIDENCE: bool = true;

/// A file entry that belongs to a duplicate group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateEntry {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub dimensions: (i32, i32),
    pub modified: Option<String>,
    pub captured_at: Option<String>,
    pub dominant_color: [u8; 3],
    pub confidence: f32,
    pub thumbnail: Option<PathBuf>,
    /// Resolution classification for the image.
    pub resolution_tier: ResolutionTier,
    /// AI moderation tier (if classification was enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moderation_tier: Option<String>,
    /// AI-generated tags (if classification was enabled, max 5).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tags: Vec<String>,
}

/// A cluster of visually identical images identified during a scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub fingerprint: u64,
    pub files: Vec<DuplicateEntry>,
}

/// Complete summary for a scan, suitable for serialisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanSummary {
    pub groups: Vec<DuplicateGroup>,
}

impl ScanSummary {
    /// Returns an iterator over groups that contain potential duplicates.
    pub fn duplicate_groups(&self) -> impl Iterator<Item = &DuplicateGroup> {
        self.groups.iter().filter(|group| group.files.len() > 1)
    }

    /// Returns an iterator over groups that need user attention:
    /// - Duplicate groups (2+ files with same fingerprint)
    /// - Singleton groups where the file has actionable resolution (Mobile or Low)
    pub fn actionable_groups(&self) -> impl Iterator<Item = &DuplicateGroup> {
        self.groups.iter().filter(|group| {
            group.files.len() > 1
                || group
                    .files
                    .first()
                    .map(|f| f.resolution_tier.is_actionable())
                    .unwrap_or(false)
        })
    }

    /// Indicates whether any groups were discovered.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

pub fn count_entries(root: &Path) -> u64 {
    WalkDir::new(root).into_iter().count() as u64
}

pub fn scan(
    root: &Path,
    config: &ScanConfig,
    progress_bar: &Arc<ProgressBar>,
    phase: Option<&Arc<Mutex<String>>>,
) -> ScanSummary {
    // Create detector with feature detection enabled if requested
    let detector_config = DetectorConfig {
        enable_feature_detection: config.enable_feature_detection,
        ..DetectorConfig::default()
    };
    let detector = DuplicateDetector::new(detector_config);
    let cache = ThumbnailCache::new(config.thumbnail_cache_root.clone())
        .ok()
        .map(Arc::new);

    // Initialize classifier if enabled
    let classifier: Option<Arc<Mutex<ImageClassifier>>> = if config.enable_classification {
        // Load classifier configuration
        let classifier_config = ClassifierConfig::load_or_default();

        // Initialize ONNX Runtime
        let ort_path = &classifier_config.ort_library;
        if let Err(e) = classifier::init_ort_runtime(ort_path) {
            progress_bar.set_message(format!("Classification disabled: {}", e));
            None
        } else {
            // Load classifier models from config
            match ImageClassifier::from_config(classifier_config) {
                Ok(c) => Some(Arc::new(Mutex::new(c))),
                Err(e) => {
                    progress_bar.set_message(format!("Classification disabled: {}", e));
                    None
                }
            }
        }
    } else {
        None
    };

    // Set initial phase
    if let Some(phase) = phase {
        if let Ok(mut p) = phase.lock() {
            *p = "Scanning files".to_string();
        }
    }

    let groups = match config.threading {
        ThreadingMode::Parallel => scan_parallel(
            root,
            config,
            progress_bar,
            phase,
            &detector,
            &cache,
            &classifier,
        ),
        ThreadingMode::Sequential => scan_sequential(
            root,
            config,
            progress_bar,
            phase,
            &detector,
            &cache,
            &classifier,
        ),
    };

    ScanSummary { groups }
}

fn scan_parallel(
    root: &Path,
    config: &ScanConfig,
    progress_bar: &Arc<ProgressBar>,
    phase: Option<&Arc<Mutex<String>>>,
    detector: &DuplicateDetector,
    cache: &Option<Arc<ThumbnailCache>>,
    classifier: &Option<Arc<Mutex<ImageClassifier>>>,
) -> Vec<DuplicateGroup> {
    let records = WalkDir::new(root)
        .into_iter()
        .par_bridge()
        .filter_map(|entry| {
            handle_entry(
                entry,
                config,
                progress_bar,
                detector,
                cache.as_deref(),
                classifier,
            )
        })
        .fold(Vec::new, |mut collection, record| {
            collection.push(record);
            collection
        })
        .reduce(Vec::new, |mut left, mut right| {
            left.append(&mut right);
            left
        });

    progress_bar.set_message("Grouping duplicates...");
    group_records(records, detector, progress_bar, phase, config)
}

fn scan_sequential(
    root: &Path,
    config: &ScanConfig,
    progress_bar: &Arc<ProgressBar>,
    phase: Option<&Arc<Mutex<String>>>,
    detector: &DuplicateDetector,
    cache: &Option<Arc<ThumbnailCache>>,
    classifier: &Option<Arc<Mutex<ImageClassifier>>>,
) -> Vec<DuplicateGroup> {
    let mut records = Vec::new();
    for entry in WalkDir::new(root) {
        if let Some(record) = handle_entry(
            entry,
            config,
            progress_bar,
            detector,
            cache.as_deref(),
            classifier,
        ) {
            records.push(record);
        }
    }
    progress_bar.set_message("Grouping duplicates...");
    group_records(records, detector, progress_bar, phase, config)
}

/// Validate and optionally rename a file to GUID format.
///
/// If GUID renaming is enabled in config, this function validates the filename
/// is a valid GUIDv4 and renames it if needed.
///
/// # Arguments
///
/// * `path` - The original file path
/// * `config` - Scanner configuration with renaming settings
/// * `progress_bar` - Progress bar for status updates
///
/// # Returns
///
/// * `Some(PathBuf)` - The final path (original or renamed)
/// * `None` - If validation or rename fails
fn validate_and_maybe_rename(
    path: PathBuf,
    config: &ScanConfig,
    progress_bar: &ProgressBar,
) -> Option<PathBuf> {
    if !config.rename_to_guid {
        return Some(path);
    }

    match ensure_guid_name(&path) {
        Ok(new_path) => {
            if new_path != path {
                progress_bar.set_message(format!(
                    "Renamed: {} -> {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    new_path.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
            Some(new_path)
        }
        Err(error) => {
            progress_bar.set_message(format!("Rename error: {}", error));
            None
        }
    }
}

/// Analyze image features including hash, resolution, and thumbnail.
///
/// Performs duplicate detection analysis and optionally generates thumbnails
/// and detects low resolution images.
///
/// # Arguments
///
/// * `path` - Path to the image file
/// * `config` - Scanner configuration
/// * `detector` - Duplicate detector for analysis
/// * `cache` - Optional thumbnail cache
/// * `progress_bar` - Progress bar for status updates
///
/// # Returns
///
/// * `Some(ImageAnalysis)` - Analysis results with metadata
/// * `None` - If analysis fails
fn analyze_image_features(
    path: &Path,
    config: &ScanConfig,
    detector: &DuplicateDetector,
    cache: Option<&ThumbnailCache>,
    progress_bar: &ProgressBar,
) -> Option<ImageAnalysis> {
    match detector.analyze(path) {
        Ok(mut analysis) => {
            // Generate thumbnail if cache is available
            if let Some(cache) = cache {
                match cache.ensure(path, analysis.features.fingerprint) {
                    Ok(thumbnail) => {
                        analysis.metadata.thumbnail = Some(thumbnail);
                    }
                    Err(error) => {
                        progress_bar.set_message(format!("Thumbnail error: {}", error));
                    }
                }
            }

            // Detect low resolution if enabled
            if config.detect_low_resolution {
                let (w, h) = analysis.metadata.dimensions;
                analysis.metadata.resolution_tier = resolution_tier(w, h);
            }

            Some(analysis)
        }
        Err(error) => {
            progress_bar.set_message(format!("Error: {}", error));
            None
        }
    }
}

/// Run AI classification (moderation + tagging) on an image.
///
/// Performs content moderation and tag generation using the provided classifier.
/// Updates the metadata in place with results.
///
/// # Arguments
///
/// * `path` - Path to the image file
/// * `classifier` - Locked image classifier
/// * `metadata` - Image metadata to update with classification results
/// * `progress_bar` - Progress bar for status updates
fn classify_and_tag(
    path: &Path,
    classifier: &mut ImageClassifier,
    metadata: &mut ImageMetadata,
    progress_bar: &ProgressBar,
) {
    // Run moderation
    match classifier.moderate(path) {
        Ok(flags) => {
            metadata.moderation_tier = Some(flags.tier.to_string());
        }
        Err(e) => {
            eprintln!(
                "Moderation classification failed for {}: {}",
                path.display(),
                e
            );
            progress_bar.set_message(format!("Moderation error: {}", e));
        }
    }

    // Run tagging (max 5 tags)
    match classifier.tag(path, MAX_TAGS_PER_IMAGE) {
        Ok(tags) => {
            metadata.tags = tags
                .into_iter()
                .map(|t| {
                    if DEBUG_SHOW_TAG_CONFIDENCE {
                        let percentage = (t.confidence * 100.0).round() as u32;
                        if percentage > 0 {
                            format!("{} ({}%)", t.label, percentage)
                        } else {
                            // For very small confidences, show decimal percentage
                            format!("{} ({:.2}%)", t.label, t.confidence * 100.0)
                        }
                    } else {
                        t.label
                    }
                })
                .collect();
        }
        Err(e) => {
            progress_bar.set_message(format!("Tagging error: {}", e));
        }
    }
}

fn handle_entry(
    entry: Result<walkdir::DirEntry, walkdir::Error>,
    config: &ScanConfig,
    progress_bar: &Arc<ProgressBar>,
    detector: &DuplicateDetector,
    cache: Option<&ThumbnailCache>,
    classifier: &Option<Arc<Mutex<ImageClassifier>>>,
) -> Option<ImageRecord> {
    progress_bar.inc(1);

    // Extract and validate entry
    let entry = match entry {
        Ok(entry) => entry,
        Err(error) => {
            progress_bar.set_message(format!("Error: {}", error));
            return None;
        }
    };

    let path = entry.path().to_path_buf();
    progress_bar.set_message(format!("Scanning: {}", path.display()));

    // Check if it's an image file
    if !path.is_file() || !has_image_extension(&path, &config.extensions) {
        return None;
    }

    // Step 1: Validate and optionally rename to GUID format
    let path = validate_and_maybe_rename(path, config, progress_bar)?;

    // Step 2: Analyze image features (hash, resolution, thumbnail)
    let mut analysis = analyze_image_features(&path, config, detector, cache, progress_bar)?;

    // Step 3: Run AI classification if enabled
    if let Some(classifier) = classifier {
        if let Ok(mut clf) = classifier.lock() {
            classify_and_tag(&path, &mut clf, &mut analysis.metadata, progress_bar);
        }
    }

    Some(ImageRecord { path, analysis })
}

fn has_image_extension(path: &Path, extensions: &[String]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let lower = ext.to_lowercase();
            extensions.iter().any(|candidate| candidate == &lower)
        })
        .unwrap_or(false)
}

fn group_records(
    records: Vec<ImageRecord>,
    detector: &DuplicateDetector,
    progress_bar: &ProgressBar,
    phase: Option<&Arc<Mutex<String>>>,
    config: &ScanConfig,
) -> Vec<DuplicateGroup> {
    // Update phase for grouping
    if let Some(phase) = phase {
        if let Ok(mut p) = phase.lock() {
            *p = "Grouping by hash".to_string();
        }
    }

    let mut groups = Vec::new();
    let mut buckets: FxHashMap<u16, Vec<usize>> = FxHashMap::default();
    for record in records {
        insert_record(&mut groups, &mut buckets, record, detector);
    }

    // Phase 2: Merge crop-related groups if feature detection is enabled
    let mut groups = if detector.config().enable_feature_detection {
        merge_crop_groups(groups, detector, progress_bar, phase)
    } else {
        groups
    };

    // Phase 3: Sort entries within each group according to preferences
    if config.prefer_display_aspect_ratios {
        for group in &mut groups {
            group.entries.sort_by(|a, b| {
                let priority_a: AspectRatioPriority =
                    get_aspect_ratio_priority(a.metadata.dimensions.0, a.metadata.dimensions.1);
                let priority_b: AspectRatioPriority =
                    get_aspect_ratio_priority(b.metadata.dimensions.0, b.metadata.dimensions.1);

                priority_b
                    .cmp(&priority_a)
                    .then_with(|| {
                        let res_a = a.metadata.dimensions.0 as u64 * a.metadata.dimensions.1 as u64;
                        let res_b = b.metadata.dimensions.0 as u64 * b.metadata.dimensions.1 as u64;
                        res_b.cmp(&res_a)
                    })
                    .then_with(|| b.metadata.size_bytes.cmp(&a.metadata.size_bytes))
            });
        }
    }

    groups
        .into_iter()
        .map(|state| DuplicateGroup {
            fingerprint: state.features.fingerprint,
            files: state
                .entries
                .into_iter()
                .map(|entry| {
                    let AnalyzedFile { path, metadata } = entry;
                    DuplicateEntry {
                        path,
                        size_bytes: metadata.size_bytes,
                        dimensions: metadata.dimensions,
                        modified: metadata.modified,
                        captured_at: metadata.captured_at,
                        dominant_color: metadata.dominant_color,
                        confidence: metadata.confidence,
                        thumbnail: metadata.thumbnail,
                        resolution_tier: metadata.resolution_tier,
                        moderation_tier: metadata.moderation_tier,
                        tags: metadata.tags,
                    }
                })
                .collect(),
        })
        .collect()
}

/// Checks if two images could potentially be crop-related based on dimensions.
/// A crop must fit within the bounds of the original image.
fn could_be_crop_related(dims_a: (i32, i32), dims_b: (i32, i32)) -> bool {
    let (w_a, h_a) = dims_a;
    let (w_b, h_b) = dims_b;

    // One could be a crop of the other if one fits inside the other
    // Allow some tolerance for slight size differences (e.g., resizing after crop)
    let tolerance = 1.1; // 10% size tolerance

    // Check if A could be a crop of B (A smaller or equal)
    let a_in_b =
        (w_a as f32) <= (w_b as f32 * tolerance) && (h_a as f32) <= (h_b as f32 * tolerance);
    // Check if B could be a crop of A (B smaller or equal)
    let b_in_a =
        (w_b as f32) <= (w_a as f32 * tolerance) && (h_b as f32) <= (h_a as f32 * tolerance);

    a_in_b || b_in_a
}

/// Merges groups that are crop-related based on ORB feature matching.
/// This is a post-processing step that compares representative features from each group.
/// Uses parallel processing for dramatic speedup on multi-core CPUs.
fn merge_crop_groups(
    mut groups: Vec<GroupState>,
    detector: &DuplicateDetector,
    progress_bar: &ProgressBar,
    phase: Option<&Arc<Mutex<String>>>,
) -> Vec<GroupState> {
    if groups.len() < 2 {
        return groups;
    }

    // Update phase for feature matching
    if let Some(phase) = phase {
        if let Ok(mut p) = phase.lock() {
            *p = "Matching crops".to_string();
        }
    }

    // Pre-filter: only compare groups that could potentially be crops
    // Build list of candidate pairs based on dimension compatibility
    let mut candidate_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..groups.len() {
        // Skip groups without visual features
        if groups[i].features.visual.is_none() {
            continue;
        }
        let dims_i = groups[i]
            .features
            .visual
            .as_ref()
            .unwrap()
            .source_dimensions;

        for j in (i + 1)..groups.len() {
            if groups[j].features.visual.is_none() {
                continue;
            }
            let dims_j = groups[j]
                .features
                .visual
                .as_ref()
                .unwrap()
                .source_dimensions;

            // Only compare if dimensions suggest possible crop relationship
            if could_be_crop_related(dims_i, dims_j) {
                candidate_pairs.push((i, j));
            }
        }
    }

    let total_comparisons = candidate_pairs.len();
    progress_bar.set_length(total_comparisons as u64);
    progress_bar.set_position(0);
    let msg = format!("Matching {} candidate pairs...", total_comparisons);
    progress_bar.set_message(msg);

    if candidate_pairs.is_empty() {
        return groups;
    }

    // Parallel comparison using rayon
    // Returns a list of (i, j) pairs that are crop matches
    let progress_counter = std::sync::atomic::AtomicU64::new(0);
    let crop_matches: Vec<(usize, usize)> = candidate_pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            // Update progress periodically
            let count = progress_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 100 == 0 {
                progress_bar.set_position(count);
            }

            // Check if these groups are crop-related using feature matching
            let match_result = detector.is_match(&groups[i].features, &groups[j].features);
            if match_result == MatchResult::CropMatch {
                Some((i, j))
            } else {
                None
            }
        })
        .collect();

    progress_bar.set_position(total_comparisons as u64);
    progress_bar.set_message("Merging matched groups...");

    // Build merge map from parallel results
    let mut merge_into: Vec<Option<usize>> = vec![None; groups.len()];
    for (i, j) in crop_matches {
        // If j is not already merged elsewhere, merge into i
        if merge_into[j].is_none() && merge_into[i].is_none() {
            merge_into[j] = Some(i);
        } else if merge_into[j].is_none() {
            // i is already merged somewhere, follow the chain
            let mut target = i;
            while let Some(t) = merge_into[target] {
                target = t;
            }
            merge_into[j] = Some(target);
        }
    }

    // Apply merges: collect entries from groups marked for merging
    // Process in reverse to avoid index invalidation issues
    for j in (0..groups.len()).rev() {
        if let Some(target) = merge_into[j] {
            // Move entries from group j to group target
            let entries_to_move = std::mem::take(&mut groups[j].entries);
            groups[target].entries.extend(entries_to_move);
        }
    }

    // Filter out empty groups (those that were merged into others)
    groups
        .into_iter()
        .filter(|g| !g.entries.is_empty())
        .collect()
}

fn insert_record(
    groups: &mut Vec<GroupState>,
    buckets: &mut FxHashMap<u16, Vec<usize>>,
    record: ImageRecord,
    detector: &DuplicateDetector,
) {
    let ImageRecord { path, analysis } = record;
    let ImageAnalysis { features, metadata } = analysis;
    let bucket = bucket_id(features.fingerprint);

    // Fast hash-based check via bucket lookup
    if let Some(indices) = buckets.get(&bucket) {
        for &index in indices {
            if detector.is_similar(&groups[index].features, &features) {
                groups[index].entries.push(AnalyzedFile {
                    path: path.clone(),
                    metadata: metadata.clone(),
                });
                return;
            }
        }
    }

    // Check all groups for hash match (different bucket but similar hash)
    for (index, group) in groups.iter().enumerate() {
        if detector.is_similar(&group.features, &features) {
            groups[index].entries.push(AnalyzedFile {
                path: path.clone(),
                metadata: metadata.clone(),
            });
            ensure_bucket_mapping(buckets, bucket, index);
            return;
        }
    }

    // No hash match found - create new group
    let index = groups.len();
    groups.push(GroupState {
        features,
        entries: vec![AnalyzedFile { path, metadata }],
    });
    ensure_bucket_mapping(buckets, bucket, index);
}

fn ensure_bucket_mapping(buckets: &mut FxHashMap<u16, Vec<usize>>, bucket: u16, index: usize) {
    let entry = buckets.entry(bucket).or_default();
    if !entry.contains(&index) {
        entry.push(index);
    }
}

fn bucket_id(fingerprint: u64) -> u16 {
    (fingerprint >> BUCKET_PREFIX_BITS) as u16
}

struct ImageRecord {
    path: PathBuf,
    analysis: ImageAnalysis,
}

struct GroupState {
    features: ImageFeatures,
    entries: Vec<AnalyzedFile>,
}

struct AnalyzedFile {
    path: PathBuf,
    metadata: ImageMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use indicatif::ProgressBar;
    use opencv::core::{self, Scalar, Vector};
    use opencv::imgcodecs;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn write_image(path: &Path, color: u8) {
        let image = core::Mat::new_rows_cols_with_default(
            64,
            64,
            core::CV_8UC3,
            Scalar::from((color as f64, color as f64, color as f64, 0.0)),
        )
        .unwrap();
        let params = Vector::<i32>::new();
        imgcodecs::imwrite(path.to_string_lossy().as_ref(), &image, &params).unwrap();
    }

    fn scan_duplicates(mode: ThreadingMode) {
        let dir = tempdir().unwrap();
        let thumb_dir = tempdir().unwrap();
        let first = dir.path().join("a.jpg");
        let second = dir.path().join("b.jpg");
        let third = dir.path().join("c.png");
        write_image(&first, 64);
        write_image(&second, 64);
        write_image(&third, 200);
        let progress = Arc::new(ProgressBar::hidden());
        let config = ScanConfig::new(vec![String::from("jpg"), String::from("png")], mode)
            .with_thumbnail_root(thumb_dir.path().to_path_buf());
        let summary = scan(dir.path(), &config, &progress, None);
        let duplicates: Vec<_> = summary.duplicate_groups().collect();
        assert_eq!(duplicates.len(), 1);
        let files = &duplicates[0].files;
        let paths: Vec<_> = files.iter().map(|entry| entry.path.clone()).collect();
        assert!(paths.contains(&first));
        assert!(paths.contains(&second));
        assert!(files.iter().all(|entry| entry.size_bytes > 0));
        assert!(files.iter().all(|entry| entry.dimensions == (64, 64)));
        assert!(files.iter().all(|entry| entry
            .thumbnail
            .as_ref()
            .map(|path| path.exists())
            .unwrap_or(false)));
        assert!(summary
            .groups
            .iter()
            .any(|group| group.files.len() == 1 && group.files[0].path == third));
    }

    #[test]
    fn scan_detects_duplicates_parallel() {
        scan_duplicates(ThreadingMode::Parallel);
    }

    #[test]
    fn scan_detects_duplicates_sequential() {
        scan_duplicates(ThreadingMode::Sequential);
    }

    #[test]
    fn count_entries_includes_root_and_files() {
        let dir = tempdir().unwrap();
        write_image(&dir.path().join("a.jpg"), 0);
        write_image(&dir.path().join("b.jpg"), 128);
        write_image(&dir.path().join("c.jpg"), 255);
        assert_eq!(count_entries(dir.path()), 4);
    }
}
