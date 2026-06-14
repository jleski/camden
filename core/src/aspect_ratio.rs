//! Utilities for calculating and evaluating image aspect ratios.

/// Priority levels used to rank display-friendly aspect ratios.
///
/// Ordering is intentional: higher variants are preferred during duplicate sorting.
/// `Uhd` is the highest priority (21:9 / 9:21 ultrawide), followed by `Wide` (between 16:9 and
/// 21:9), then `Fhd` (16:9 / 9:16), `Standard` (other common ratios), and `NonStandard`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AspectRatioPriority {
    NonStandard,
    Standard,
    Fhd,
    /// Ratios strictly between 16:9 and 21:9 (e.g., 2:1, 1.85:1).
    Wide,
    Uhd,
}

/// A list of standard display and photographic aspect ratios.
/// Ratios are represented as `(width, height)`.
const STANDARD_ASPECT_RATIOS: &[(f32, f32)] = &[
    (16.0, 9.0),  // Widescreen video
    (4.0, 3.0),   // Standard TV / Monitor
    (3.0, 2.0),   // 35mm film
    (1.0, 1.0),   // Square
    (16.0, 10.0), // Widescreen computer displays
    (5.0, 4.0),   // Common for larger format photography
    (21.0, 9.0),  // Ultrawide cinema
];

/// The tolerance to use when comparing aspect ratios.
/// An aspect ratio is considered "standard" if it is within this tolerance
/// of a known standard ratio.
const ASPECT_RATIO_TOLERANCE: f32 = 0.05;

const FHD_ASPECT_RATIO: (f32, f32) = (16.0, 9.0);
const UHD_ASPECT_RATIO: (f32, f32) = (21.0, 9.0);

fn calculate_normalized_ratio(width: f32, height: f32) -> f32 {
    let larger = width.max(height);
    let smaller = width.min(height);
    larger / smaller
}

fn is_ratio_match(actual_ratio: f32, expected_ratio: (f32, f32)) -> bool {
    let expected_normalized_ratio = calculate_normalized_ratio(expected_ratio.0, expected_ratio.1);
    (actual_ratio - expected_normalized_ratio).abs() < ASPECT_RATIO_TOLERANCE
}

/// Returns the ranking priority for an image's aspect ratio.
///
/// The ratio is normalized as `max(width, height) / min(width, height)` so portrait and
/// landscape forms of the same ratio are treated equally.
///
/// Priority order (highest first): `Uhd` (21:9) > `Wide` (between 16:9 and 21:9) >
/// `Fhd` (16:9) > `Standard` (other common ratios) > `NonStandard`.
///
/// # Examples
///
/// ```
/// use camden_core::aspect_ratio::{get_aspect_ratio_priority, AspectRatioPriority};
///
/// assert_eq!(get_aspect_ratio_priority(2560, 1080), AspectRatioPriority::Uhd);
/// assert_eq!(get_aspect_ratio_priority(1080, 1920), AspectRatioPriority::Fhd);
/// assert_eq!(get_aspect_ratio_priority(2000, 1000), AspectRatioPriority::Wide);
/// assert_eq!(get_aspect_ratio_priority(1920, 1080), AspectRatioPriority::Fhd);
/// ```
pub fn get_aspect_ratio_priority(width: i32, height: i32) -> AspectRatioPriority {
    if width <= 0 || height <= 0 {
        return AspectRatioPriority::NonStandard;
    }

    let normalized_ratio = calculate_normalized_ratio(width as f32, height as f32);

    if is_ratio_match(normalized_ratio, UHD_ASPECT_RATIO) {
        return AspectRatioPriority::Uhd;
    }

    if is_ratio_match(normalized_ratio, FHD_ASPECT_RATIO) {
        return AspectRatioPriority::Fhd;
    }

    // Wide: strictly between FHD (16:9 ≈ 1.778) and UHD (21:9 ≈ 2.333), outside both tolerance bands.
    let fhd_ratio = calculate_normalized_ratio(FHD_ASPECT_RATIO.0, FHD_ASPECT_RATIO.1);
    let uhd_ratio = calculate_normalized_ratio(UHD_ASPECT_RATIO.0, UHD_ASPECT_RATIO.1);
    let fhd_upper = fhd_ratio + ASPECT_RATIO_TOLERANCE;
    let uhd_lower = uhd_ratio - ASPECT_RATIO_TOLERANCE;
    if normalized_ratio > fhd_upper && normalized_ratio < uhd_lower {
        return AspectRatioPriority::Wide;
    }

    for &(standard_width, standard_height) in STANDARD_ASPECT_RATIOS {
        if is_ratio_match(normalized_ratio, (standard_width, standard_height)) {
            return AspectRatioPriority::Standard;
        }
    }

    AspectRatioPriority::NonStandard
}

/// Checks if a given aspect ratio is close to one of the standard ratios.
///
/// # Arguments
///
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Returns
///
/// `true` if the aspect ratio is considered standard, `false` otherwise.
pub fn is_standard_aspect_ratio(width: i32, height: i32) -> bool {
    get_aspect_ratio_priority(width, height) >= AspectRatioPriority::Standard
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_aspect_ratio_priority() {
        assert_eq!(
            get_aspect_ratio_priority(2560, 1080),
            AspectRatioPriority::Uhd
        );
        assert_eq!(
            get_aspect_ratio_priority(1920, 1080),
            AspectRatioPriority::Fhd
        );
        assert_eq!(
            get_aspect_ratio_priority(1080, 1920),
            AspectRatioPriority::Fhd
        );
        assert_eq!(
            get_aspect_ratio_priority(1000, 1000),
            AspectRatioPriority::Standard
        );
        // 1000x999 ≈ 1.001:1, which is within tolerance of 1:1 → Standard
        assert_eq!(
            get_aspect_ratio_priority(1000, 999),
            AspectRatioPriority::Standard
        );
    }

    #[test]
    fn wide_tier_detected() {
        // 2000x1000 = 2.0:1 — strictly between 16:9 (1.778) and 21:9 (2.333)
        assert_eq!(
            get_aspect_ratio_priority(2000, 1000),
            AspectRatioPriority::Wide
        );
        // Portrait equivalent
        assert_eq!(
            get_aspect_ratio_priority(1000, 2000),
            AspectRatioPriority::Wide
        );
    }

    #[test]
    fn uhd_detected() {
        // Landscape 21:9
        assert_eq!(
            get_aspect_ratio_priority(2560, 1080),
            AspectRatioPriority::Uhd
        );
        // Portrait 9:21
        assert_eq!(
            get_aspect_ratio_priority(1080, 2560),
            AspectRatioPriority::Uhd
        );
    }

    #[test]
    fn fhd_detected() {
        assert_eq!(
            get_aspect_ratio_priority(1920, 1080),
            AspectRatioPriority::Fhd
        );
        assert_eq!(
            get_aspect_ratio_priority(1080, 1920),
            AspectRatioPriority::Fhd
        );
    }

    #[test]
    fn test_is_standard_aspect_ratio() {
        // Exact 16:9
        assert!(is_standard_aspect_ratio(1920, 1080));
        // Close to 16:9
        assert!(is_standard_aspect_ratio(1921, 1080));
        // 1000x999 ≈ 1.001:1, within tolerance of 1:1 → is standard
        assert!(is_standard_aspect_ratio(1000, 999));
        // Exact 4:3
        assert!(is_standard_aspect_ratio(1024, 768));
        // Exact 3:2
        assert!(is_standard_aspect_ratio(1080, 720));
        // Exact 1:1
        assert!(is_standard_aspect_ratio(1000, 1000));
        // Portrait orientation (9:16) should also be standard
        assert!(is_standard_aspect_ratio(1080, 1920));
        // Zero width or height
        assert!(!is_standard_aspect_ratio(0, 1080));
        assert!(!is_standard_aspect_ratio(1920, 0));
        // Wide is >= Standard so it's also "standard"
        assert!(is_standard_aspect_ratio(2000, 1000));
        // UHD is >= Standard so it's also "standard"
        assert!(is_standard_aspect_ratio(2560, 1080));
    }
}
