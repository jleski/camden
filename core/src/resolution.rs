//! Low-resolution image detection utilities.
//!
//! Resolution is determined by a single dimension only:
//! - Landscape/square: only **width** matters (height is irrelevant)
//! - Portrait: only **height** matters (width is irrelevant)

use serde::{Deserialize, Serialize};

/// Minimum width for landscape images to be considered "high resolution".
pub const MIN_LANDSCAPE_WIDTH: i32 = 1200;

/// Minimum height for portrait images to be considered "high resolution" (desktop).
pub const MIN_PORTRAIT_HEIGHT_DESKTOP: i32 = 1200;

/// Minimum height for portrait images to be usable on mobile screens.
pub const MIN_PORTRAIT_HEIGHT_MOBILE: i32 = 850;

/// Default minimum longest-edge length (in pixels) used by [`LowResolutionConfig::default`].
pub const DEFAULT_MIN_LONGEST_EDGE: i32 = 1900;

/// User-configurable rules for flagging "low resolution" images.
///
/// Unlike the legacy [`resolution_tier`] function (which uses fixed 1200/850px
/// thresholds and always distinguishes a `Mobile` tier), this config classifies
/// purely by the image's **longest edge** against a single configurable
/// threshold, and lets the user opt in/out of checking landscape and/or
/// portrait images independently.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LowResolutionConfig {
    /// Minimum longest-edge length (in pixels) for an image to be considered
    /// high resolution. Images whose longest edge is smaller are flagged "Low".
    pub min_longest_edge: i32,
    /// When true, landscape and square images are checked against `min_longest_edge`.
    pub check_landscape: bool,
    /// When true, portrait images are checked against `min_longest_edge`.
    pub check_portrait: bool,
}

impl Default for LowResolutionConfig {
    fn default() -> Self {
        Self {
            min_longest_edge: DEFAULT_MIN_LONGEST_EDGE,
            check_landscape: true,
            check_portrait: true,
        }
    }
}

impl LowResolutionConfig {
    /// Classifies an image as `Low` or `High` according to this configuration.
    ///
    /// Orientations that are not enabled for checking (`check_landscape` /
    /// `check_portrait`) are always classified as `High` (i.e. not flagged),
    /// regardless of their actual dimensions. The `Mobile` tier is never
    /// produced by this function; it exists solely for the legacy
    /// [`resolution_tier`] thresholds.
    ///
    /// # Examples
    ///
    /// ```
    /// use camden_core::resolution::LowResolutionConfig;
    /// use camden_core::resolution::ResolutionTier;
    ///
    /// let config = LowResolutionConfig::default();
    /// assert_eq!(config.classify(2560, 1440), ResolutionTier::High);
    /// assert_eq!(config.classify(1280, 720), ResolutionTier::Low);
    /// ```
    pub fn classify(&self, width: i32, height: i32) -> ResolutionTier {
        let longest_edge = width.max(height);
        let is_portrait = height > width;

        if is_portrait && !self.check_portrait {
            return ResolutionTier::High;
        }
        if !is_portrait && !self.check_landscape {
            return ResolutionTier::High;
        }

        if longest_edge >= self.min_longest_edge {
            ResolutionTier::High
        } else {
            ResolutionTier::Low
        }
    }
}

/// Resolution classification for an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResolutionTier {
    /// Image meets desktop resolution requirements.
    #[default]
    High,
    /// Portrait image suitable for mobile but not desktop (850-1199px height).
    Mobile,
    /// Image is too low resolution even for mobile use.
    Low,
}

impl ResolutionTier {
    /// Returns true if the image should be flagged in the UI.
    pub fn is_actionable(self) -> bool {
        matches!(self, ResolutionTier::Mobile | ResolutionTier::Low)
    }

    /// Returns true if the image should be pre-selected for moving.
    pub fn should_preselect(self) -> bool {
        matches!(self, ResolutionTier::Low)
    }
}

/// Determines the resolution tier for an image based on its dimensions.
///
/// - **Landscape/square** (width >= height):
///   - High if width >= 1200, else Low
/// - **Portrait** (height > width):
///   - High if height >= 1200
///   - Mobile if height >= 850 (usable on mobile screens)
///   - Low if height < 850
pub fn resolution_tier(width: i32, height: i32) -> ResolutionTier {
    if width >= height {
        // Landscape or square: single tier based on width
        if width >= MIN_LANDSCAPE_WIDTH {
            ResolutionTier::High
        } else {
            ResolutionTier::Low
        }
    } else {
        // Portrait: two-tier based on height
        if height >= MIN_PORTRAIT_HEIGHT_DESKTOP {
            ResolutionTier::High
        } else if height >= MIN_PORTRAIT_HEIGHT_MOBILE {
            ResolutionTier::Mobile
        } else {
            ResolutionTier::Low
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landscape_high_resolution() {
        assert_eq!(resolution_tier(1200, 800), ResolutionTier::High);
        assert_eq!(resolution_tier(1920, 1080), ResolutionTier::High);
        assert_eq!(resolution_tier(3840, 2160), ResolutionTier::High);
    }

    #[test]
    fn landscape_low_resolution() {
        assert_eq!(resolution_tier(1199, 800), ResolutionTier::Low);
        assert_eq!(resolution_tier(1080, 720), ResolutionTier::Low);
        assert_eq!(resolution_tier(800, 600), ResolutionTier::Low);
    }

    #[test]
    fn landscape_ignores_height() {
        // Width is 1200 (OK), height varies - all should be high
        assert_eq!(resolution_tier(1200, 100), ResolutionTier::High);
        assert_eq!(resolution_tier(1200, 500), ResolutionTier::High);
        assert_eq!(resolution_tier(1200, 800), ResolutionTier::High);
        // Width is 1100 (low), height varies - all should be low
        assert_eq!(resolution_tier(1100, 100), ResolutionTier::Low);
        assert_eq!(resolution_tier(1100, 720), ResolutionTier::Low);
    }

    #[test]
    fn portrait_high_resolution() {
        assert_eq!(resolution_tier(800, 1200), ResolutionTier::High);
        assert_eq!(resolution_tier(1080, 1920), ResolutionTier::High);
    }

    #[test]
    fn portrait_mobile_resolution() {
        assert_eq!(resolution_tier(600, 850), ResolutionTier::Mobile);
        assert_eq!(resolution_tier(720, 1000), ResolutionTier::Mobile);
        assert_eq!(resolution_tier(800, 1199), ResolutionTier::Mobile);
    }

    #[test]
    fn portrait_low_resolution() {
        assert_eq!(resolution_tier(400, 849), ResolutionTier::Low);
        assert_eq!(resolution_tier(480, 640), ResolutionTier::Low);
    }

    #[test]
    fn portrait_ignores_width() {
        // Height is 1200 (high), width varies
        assert_eq!(resolution_tier(100, 1200), ResolutionTier::High);
        assert_eq!(resolution_tier(800, 1200), ResolutionTier::High);
        // Height is 1000 (mobile), width varies
        assert_eq!(resolution_tier(100, 1000), ResolutionTier::Mobile);
        assert_eq!(resolution_tier(600, 1000), ResolutionTier::Mobile);
        // Height is 800 (low), width varies
        assert_eq!(resolution_tier(100, 800), ResolutionTier::Low);
        assert_eq!(resolution_tier(500, 800), ResolutionTier::Low);
    }

    #[test]
    fn square_uses_landscape_rule() {
        assert_eq!(resolution_tier(1000, 1000), ResolutionTier::Low);
        assert_eq!(resolution_tier(1200, 1200), ResolutionTier::High);
    }

    #[test]
    fn tier_actionable() {
        assert!(!ResolutionTier::High.is_actionable());
        assert!(ResolutionTier::Mobile.is_actionable());
        assert!(ResolutionTier::Low.is_actionable());
    }

    #[test]
    fn tier_preselect() {
        assert!(!ResolutionTier::High.should_preselect());
        assert!(!ResolutionTier::Mobile.should_preselect());
        assert!(ResolutionTier::Low.should_preselect());
    }

    #[test]
    fn low_resolution_config_default_is_1900_both_orientations() {
        let config = LowResolutionConfig::default();
        assert_eq!(config.min_longest_edge, 1900);
        assert!(config.check_landscape);
        assert!(config.check_portrait);
    }

    #[test]
    fn low_resolution_config_classifies_by_longest_edge() {
        let config = LowResolutionConfig::default();
        // Landscape: longest edge is width.
        assert_eq!(config.classify(1920, 1080), ResolutionTier::High);
        assert_eq!(config.classify(2560, 1440), ResolutionTier::High);
        assert_eq!(config.classify(1900, 1080), ResolutionTier::High); // exactly at threshold
        assert_eq!(config.classify(1899, 1080), ResolutionTier::Low);
        // Portrait: longest edge is height.
        assert_eq!(config.classify(1080, 1920), ResolutionTier::High);
        assert_eq!(config.classify(1080, 1899), ResolutionTier::Low);
    }

    #[test]
    fn low_resolution_config_respects_custom_threshold() {
        let config = LowResolutionConfig {
            min_longest_edge: 3000,
            check_landscape: true,
            check_portrait: true,
        };
        assert_eq!(config.classify(2560, 1440), ResolutionTier::Low);
        assert_eq!(config.classify(3840, 2160), ResolutionTier::High);
    }

    #[test]
    fn low_resolution_config_can_ignore_landscape() {
        let config = LowResolutionConfig {
            min_longest_edge: 1900,
            check_landscape: false,
            check_portrait: true,
        };
        // Landscape images are never flagged when check_landscape is false.
        assert_eq!(config.classify(800, 600), ResolutionTier::High);
        // Portrait images are still checked.
        assert_eq!(config.classify(600, 800), ResolutionTier::Low);
    }

    #[test]
    fn low_resolution_config_can_ignore_portrait() {
        let config = LowResolutionConfig {
            min_longest_edge: 1900,
            check_landscape: true,
            check_portrait: false,
        };
        // Portrait images are never flagged when check_portrait is false.
        assert_eq!(config.classify(600, 800), ResolutionTier::High);
        // Landscape images are still checked.
        assert_eq!(config.classify(800, 600), ResolutionTier::Low);
    }

    #[test]
    fn low_resolution_config_square_uses_landscape_path() {
        let config = LowResolutionConfig {
            min_longest_edge: 1900,
            check_landscape: false,
            check_portrait: true,
        };
        // height > width is false for square, so it's treated as landscape.
        assert_eq!(config.classify(1000, 1000), ResolutionTier::High);
    }
}
