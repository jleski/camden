//! Utilities for calculating and evaluating image aspect ratios.

/// Priority levels used to rank display-friendly aspect ratios.
///
/// Ordering is intentional: higher variants are preferred during duplicate sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AspectRatioPriority {
    NonStandard,
    Standard,
    Fhd,
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
/// # Examples
///
/// ```
/// use camden_core::aspect_ratio::{get_aspect_ratio_priority, AspectRatioPriority};
///
/// assert_eq!(
///     get_aspect_ratio_priority(2560, 1080),
///     AspectRatioPriority::Uhd
/// );
/// assert_eq!(
///     get_aspect_ratio_priority(1080, 1920),
///     AspectRatioPriority::Fhd
/// );
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
        assert_eq!(
            get_aspect_ratio_priority(1000, 999),
            AspectRatioPriority::NonStandard
        );
    }

    #[test]
    fn test_is_standard_aspect_ratio() {
        // Exact 16:9
        assert!(is_standard_aspect_ratio(1920, 1080));
        // Close to 16:9
        assert!(is_standard_aspect_ratio(1921, 1080));
        // Not standard
        assert!(!is_standard_aspect_ratio(1000, 999));
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
    }
}
