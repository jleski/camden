//! Single source of truth for keeper-selection logic.
//!
//! A "keeper" is the one file in a duplicate group that stays on disk; all other files
//! in the group are pre-selected for archival. This module provides a deterministic,
//! testable comparator and index-selector used by both the CLI and the GUI so the two
//! interfaces can never diverge again.
//!
//! # Selection rules
//!
//! When `prefer_ultrawide_aspect_ratios` is **true**:
//! 1. Aspect-ratio priority (highest wins): `Uhd` > `Wide` > `Fhd` > `Standard` > `NonStandard`.
//! 2. Larger `size_bytes` breaks ties.
//! 3. Newer date (RFC3339 string, `Some` > `None`, lexical comparison) breaks further ties.
//! 4. Lexicographically smaller path is kept (matches legacy GUI tiebreak).
//!
//! When `prefer_ultrawide_aspect_ratios` is **false** (historical default):
//! 1. Larger `size_bytes`.
//! 2. Newer date.
//! 3. Lexicographically smaller path.

use crate::aspect_ratio::get_aspect_ratio_priority;
use std::cmp::Ordering;

/// Options that control which file in a group is selected as the keeper.
pub struct KeeperPreferences {
    /// When `true`, the keeper is the widest (ultrawide-first) image regardless of size.
    /// When `false`, the keeper is the largest file (historical default).
    pub prefer_ultrawide_aspect_ratios: bool,
}

/// Lightweight description of one file in a duplicate group.
///
/// All fields are borrowed from the caller's data so no allocation is needed.
#[derive(Clone, Copy)]
pub struct KeeperCandidate<'a> {
    pub width: i32,
    pub height: i32,
    pub size_bytes: u64,
    /// Captured-at or modified date as an RFC3339 string, if available.
    pub date: Option<&'a str>,
    pub path: &'a std::path::Path,
}

/// Ordering where the **keeper** is the **maximum** element.
///
/// Pass to `Iterator::max_by` or `slice::sort_by` (descending) to put the keeper first.
///
/// ```
/// use std::path::Path;
/// use camden_core::keeper::{KeeperCandidate, KeeperPreferences, keeper_ordering};
///
/// let prefs = KeeperPreferences { prefer_ultrawide_aspect_ratios: false };
/// let small = KeeperCandidate { width: 1920, height: 1080, size_bytes: 100,
///                               date: None, path: Path::new("a.jpg") };
/// let large = KeeperCandidate { width: 1920, height: 1080, size_bytes: 200,
///                               date: None, path: Path::new("b.jpg") };
/// assert_eq!(keeper_ordering(&large, &small, &prefs), std::cmp::Ordering::Greater);
/// ```
pub fn keeper_ordering(
    a: &KeeperCandidate,
    b: &KeeperCandidate,
    prefs: &KeeperPreferences,
) -> Ordering {
    if prefs.prefer_ultrawide_aspect_ratios {
        let priority_a = get_aspect_ratio_priority(a.width, a.height);
        let priority_b = get_aspect_ratio_priority(b.width, b.height);
        priority_a
            .cmp(&priority_b)
            .then_with(|| a.size_bytes.cmp(&b.size_bytes))
            .then_with(|| a.date.cmp(&b.date))
            .then_with(|| b.path.cmp(a.path))
    } else {
        a.size_bytes
            .cmp(&b.size_bytes)
            .then_with(|| a.date.cmp(&b.date))
            .then_with(|| b.path.cmp(a.path))
    }
}

/// Returns the index of the file to **keep**; every other file should be archived.
///
/// The keeper is the candidate with the highest `keeper_ordering` value.
/// Returns `0` for an empty slice (safe default).
///
/// ```
/// use std::path::Path;
/// use camden_core::keeper::{KeeperCandidate, KeeperPreferences, select_keeper_index};
///
/// let prefs = KeeperPreferences { prefer_ultrawide_aspect_ratios: true };
/// let fhd  = KeeperCandidate { width: 1920, height: 1080, size_bytes: 500,
///                               date: None, path: Path::new("fhd.jpg") };
/// let uhd  = KeeperCandidate { width: 2560, height: 1080, size_bytes: 100,
///                               date: None, path: Path::new("uhd.jpg") };
/// // Ultrawide wins even though its file is smaller.
/// assert_eq!(select_keeper_index(&[fhd, uhd], &prefs), 1);
/// ```
pub fn select_keeper_index(candidates: &[KeeperCandidate], prefs: &KeeperPreferences) -> usize {
    if candidates.is_empty() {
        return 0;
    }
    candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| keeper_ordering(a, b, prefs))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn off() -> KeeperPreferences {
        KeeperPreferences {
            prefer_ultrawide_aspect_ratios: false,
        }
    }

    fn on() -> KeeperPreferences {
        KeeperPreferences {
            prefer_ultrawide_aspect_ratios: true,
        }
    }

    fn candidate<'a>(
        width: i32,
        height: i32,
        size: u64,
        date: Option<&'a str>,
        path: &'a str,
    ) -> KeeperCandidate<'a> {
        KeeperCandidate {
            width,
            height,
            size_bytes: size,
            date,
            path: Path::new(path),
        }
    }

    // ── Ported from camden-frontend/src/main.rs (prefer OFF) ─────────────────

    #[test]
    fn keep_largest() {
        let files = [
            candidate(1920, 1080, 100, Some("2023-01-01"), "small.jpg"),
            candidate(1920, 1080, 200, Some("2023-01-01"), "large.jpg"),
        ];
        assert_eq!(select_keeper_index(&files, &off()), 1);
    }

    #[test]
    fn keep_newest_when_size_same() {
        let files = [
            candidate(1920, 1080, 100, Some("2023-01-01"), "old.jpg"),
            candidate(1920, 1080, 100, Some("2023-01-02"), "new.jpg"),
        ];
        assert_eq!(select_keeper_index(&files, &off()), 1);
    }

    #[test]
    fn keep_alphabetical_when_size_and_date_same() {
        // Legacy GUI used `b.path.cmp(&a.path)` in max_by, meaning the lexicographically
        // *smaller* path wins (because max_by picks the one cmp says is "Greater", and
        // `b.cmp(a)` is Greater when b > a, i.e. when a < b).
        let files = [
            candidate(1920, 1080, 100, Some("2023-01-01"), "a.jpg"),
            candidate(1920, 1080, 100, Some("2023-01-01"), "b.jpg"),
        ];
        assert_eq!(select_keeper_index(&files, &off()), 0); // a.jpg kept
    }

    #[test]
    fn keep_largest_ignores_date() {
        let files = [
            candidate(1920, 1080, 100, Some("2023-01-02"), "small_new.jpg"),
            candidate(1920, 1080, 200, Some("2023-01-01"), "large_old.jpg"),
        ];
        assert_eq!(select_keeper_index(&files, &off()), 1);
    }

    #[test]
    fn handle_missing_dates() {
        // Some("...") > None for Option ordering, so the file with a date is newer.
        let files = [
            candidate(1920, 1080, 100, None, "no_date.jpg"),
            candidate(1920, 1080, 100, Some("2023-01-01"), "with_date.jpg"),
        ];
        assert_eq!(select_keeper_index(&files, &off()), 1);
    }

    // ── New tests for prefer ON ───────────────────────────────────────────────

    #[test]
    fn keeps_ultrawide_over_larger_fhd_when_preferred() {
        // 21:9 ultrawide (small file) should beat 16:9 (large file) when flag is ON.
        let files = [
            candidate(1920, 1080, 500, Some("2023-01-01"), "fhd.jpg"), // index 0 — FHD, larger
            candidate(2560, 1080, 100, Some("2023-01-01"), "uhd.jpg"), // index 1 — UHD, smaller
        ];
        assert_eq!(select_keeper_index(&files, &on()), 1);
    }

    #[test]
    fn keeps_portrait_ultrawide_over_larger_fhd_portrait() {
        // 9:21 portrait ultrawide (small) vs 9:16 portrait FHD (large).
        let files = [
            candidate(1080, 1920, 500, Some("2023-01-01"), "portrait_fhd.jpg"), // index 0 — Fhd
            candidate(1080, 2560, 100, Some("2023-01-01"), "portrait_uhd.jpg"), // index 1 — Uhd
        ];
        assert_eq!(select_keeper_index(&files, &on()), 1);
    }

    #[test]
    fn keeps_fhd_over_standard_when_preferred() {
        // No ultrawide candidate: 16:9 beats 4:3.
        let files = [
            candidate(1024, 768, 500, Some("2023-01-01"), "std.jpg"), // Standard (4:3)
            candidate(1920, 1080, 100, Some("2023-01-01"), "fhd.jpg"), // Fhd (16:9)
        ];
        assert_eq!(select_keeper_index(&files, &on()), 1);
    }

    #[test]
    fn keeps_wide_over_fhd_when_preferred() {
        // 2:1 Wide beats 16:9 Fhd when ultrawide preference is ON.
        let files = [
            candidate(1920, 1080, 500, Some("2023-01-01"), "fhd.jpg"), // Fhd
            candidate(2000, 1000, 100, Some("2023-01-01"), "wide.jpg"), // Wide (2:1)
        ];
        assert_eq!(select_keeper_index(&files, &on()), 1);
    }

    #[test]
    fn empty_slice_returns_zero() {
        assert_eq!(select_keeper_index(&[], &off()), 0);
        assert_eq!(select_keeper_index(&[], &on()), 0);
    }
}
