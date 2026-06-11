//! Core duplicate detection engine for Camden.
//!
//! This crate exposes high-level scanning, reporting, and file-move
//! operations used by both the CLI and future preview UI. The public API
//! focuses on data-transfer objects (`ScanSummary`, `DuplicateGroup`,
//! `DuplicateEntry`) that are serialisable for downstream consumers.

pub mod aspect_ratio;
#[cfg(feature = "classification")]
pub mod classifier;
pub mod detector;
pub mod operations;
pub mod progress;
pub mod rename;
pub mod reporting;
pub mod resolution;
pub mod scanner;
pub mod snapshot;
pub mod thumbnails;

pub use operations::{move_duplicates, move_paths, MoveError, MoveStats};
pub use rename::{ensure_guid_name, is_guid_named, RenameError};
pub use reporting::{print_duplicates, write_classification_report, write_json};
pub use resolution::{
    resolution_tier, ResolutionTier, MIN_LANDSCAPE_WIDTH, MIN_PORTRAIT_HEIGHT_DESKTOP,
    MIN_PORTRAIT_HEIGHT_MOBILE,
};
pub use scanner::{
    count_entries, scan, DuplicateEntry, DuplicateGroup, ScanConfig, ScanSummary, ThreadingMode,
    MAX_TAGS_PER_IMAGE,
};
pub use snapshot::{
    create_snapshot, default_snapshot_path, read_snapshot, write_snapshot, ScanSnapshot,
    SnapshotError,
};
pub use thumbnails::{ThumbnailCache, ThumbnailError};

#[cfg(feature = "classification")]
pub use classifier::{
    default_ort_dylib_path, init_ort_runtime, ClassificationResult, ClassifierError,
    ImageClassifier, ImageTag, ModelPaths, ModerationCategories, ModerationFlags, ModerationTier,
    NsfwClassifier, TagCategory, TaggingClassifier,
};
