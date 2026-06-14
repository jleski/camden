use camden_core::ThreadingMode;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Scan(CliConfig),
    Preview(PreviewConfig),
}

#[derive(Debug, PartialEq, Eq)]
pub struct CliConfig {
    pub root: PathBuf,
    pub target: Option<PathBuf>,
    pub threading: ThreadingMode,
    pub extensions: Vec<String>,
    pub rename_to_guid: bool,
    pub detect_low_resolution: bool,
    pub enable_classification: bool,
    pub enable_feature_detection: bool,
    pub prefer_ultrawide_aspect_ratios: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct PreviewConfig {
    pub root: PathBuf,
    pub output: PathBuf,
    pub threading: ThreadingMode,
    pub extensions: Vec<String>,
    pub thumbnail_root: Option<PathBuf>,
    pub rename_to_guid: bool,
    pub detect_low_resolution: bool,
    pub enable_classification: bool,
    pub enable_feature_detection: bool,
    pub prefer_ultrawide_aspect_ratios: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CliError {
    MissingRoot,
    MissingOutput,
    InvalidFlag(String),
    Help,
    Version,
}

impl Command {
    pub fn from_env() -> Result<Self, CliError> {
        Self::from_iter(env::args().skip(1))
    }

    pub fn from_iter<I>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut args = args.into_iter();
        match args.next() {
            Some(first) if first == "--version" || first == "-V" => Err(CliError::Version),
            Some(first) if first == "--help" || first == "-h" => Err(CliError::Help),
            Some(first) if first == "preview-scan" => {
                PreviewConfig::parse(args).map(Command::Preview)
            }
            Some(first) => {
                let mut rest = vec![first];
                rest.extend(args);
                CliConfig::from_iter(rest).map(Command::Scan)
            }
            None => Err(CliError::Help),
        }
    }
}

impl CliConfig {
    pub fn from_iter<I>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = String>,
    {
        Self::parse(args.into_iter())
    }

    fn parse<I>(mut args: I) -> Result<Self, CliError>
    where
        I: Iterator<Item = String>,
    {
        let mut root: Option<PathBuf> = None;
        let mut target: Option<PathBuf> = None;
        let mut threading = ThreadingMode::Parallel;
        let mut rename_to_guid = false;
        let mut detect_low_resolution = false;
        let mut enable_classification = false;
        let mut enable_feature_detection = false;
        let mut prefer_ultrawide_aspect_ratios = false;

        for arg in args.by_ref() {
            if arg.starts_with("--") {
                if arg == "--no-thread" {
                    threading = ThreadingMode::Sequential;
                    continue;
                }
                if arg == "--rename-to-guid" {
                    rename_to_guid = true;
                    continue;
                }
                if arg == "--detect-low-resolution" {
                    detect_low_resolution = true;
                    continue;
                }
                if arg == "--enable-classification" || arg == "--classify" {
                    enable_classification = true;
                    continue;
                }
                if arg == "--enable-feature-detection" || arg == "--feature-detection" {
                    enable_feature_detection = true;
                    continue;
                }
                if arg == "--prefer-ultrawide-aspect-ratios"
                    || arg == "--prefer-display-aspect-ratios"
                {
                    prefer_ultrawide_aspect_ratios = true;
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--target=") {
                    target = Some(PathBuf::from(value));
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--root=") {
                    root = Some(PathBuf::from(value));
                    continue;
                }
                return Err(CliError::InvalidFlag(arg));
            }

            if root.is_none() {
                root = Some(PathBuf::from(&arg));
                continue;
            }

            if target.is_none() {
                target = Some(PathBuf::from(&arg));
                continue;
            }

            return Err(CliError::InvalidFlag(arg));
        }

        let root = root.ok_or(CliError::MissingRoot)?;

        Ok(Self {
            root,
            target,
            threading,
            extensions: default_extensions(),
            rename_to_guid,
            detect_low_resolution,
            enable_classification,
            enable_feature_detection,
            prefer_ultrawide_aspect_ratios,
        })
    }
}

impl PreviewConfig {
    fn parse<I>(mut args: I) -> Result<Self, CliError>
    where
        I: Iterator<Item = String>,
    {
        let mut root: Option<PathBuf> = None;
        let mut output: Option<PathBuf> = None;
        let mut thumbnail_root: Option<PathBuf> = None;
        let mut threading = ThreadingMode::Parallel;
        let mut rename_to_guid = false;
        let mut detect_low_resolution = false;
        let mut enable_classification = false;
        let mut enable_feature_detection = false;
        let mut prefer_ultrawide_aspect_ratios = false;

        for arg in args.by_ref() {
            if arg.starts_with("--") {
                if arg == "--no-thread" {
                    threading = ThreadingMode::Sequential;
                    continue;
                }
                if arg == "--rename-to-guid" {
                    rename_to_guid = true;
                    continue;
                }
                if arg == "--detect-low-resolution" {
                    detect_low_resolution = true;
                    continue;
                }
                if arg == "--enable-classification" || arg == "--classify" {
                    enable_classification = true;
                    continue;
                }
                if arg == "--enable-feature-detection" || arg == "--feature-detection" {
                    enable_feature_detection = true;
                    continue;
                }
                if arg == "--prefer-ultrawide-aspect-ratios"
                    || arg == "--prefer-display-aspect-ratios"
                {
                    prefer_ultrawide_aspect_ratios = true;
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--root=") {
                    root = Some(PathBuf::from(value));
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--output=") {
                    output = Some(PathBuf::from(value));
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--thumbnail-root=") {
                    thumbnail_root = Some(PathBuf::from(value));
                    continue;
                }
                return Err(CliError::InvalidFlag(arg));
            }

            if root.is_none() {
                root = Some(PathBuf::from(&arg));
                continue;
            }

            if output.is_none() {
                output = Some(PathBuf::from(&arg));
                continue;
            }

            return Err(CliError::InvalidFlag(arg));
        }

        let root = root.ok_or(CliError::MissingRoot)?;
        let output = output
            .or_else(camden_core::default_snapshot_path)
            .ok_or(CliError::MissingOutput)?;

        Ok(Self {
            root,
            output,
            threading,
            extensions: default_extensions(),
            thumbnail_root,
            rename_to_guid,
            detect_low_resolution,
            enable_classification,
            enable_feature_detection,
            prefer_ultrawide_aspect_ratios,
        })
    }

    pub fn thumbnail_root(&self) -> Option<&Path> {
        self.thumbnail_root.as_deref()
    }
}

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRoot => write!(f, "root directory argument is required"),
            Self::MissingOutput => write!(f, "snapshot output path is required"),
            Self::InvalidFlag(flag) => write!(f, "unrecognized argument: {}", flag),
            Self::Help => write!(f, "{}", help_text()),
            Self::Version => write!(f, "{}", env!("CAMDEN_VERSION_FULL")),
        }
    }
}

fn help_text() -> String {
    format!(
        r#"Camden - Image Duplicate Finder {}

USAGE:
    camden <ROOT> [TARGET] [OPTIONS]
    camden preview-scan <ROOT> [OPTIONS]

COMMANDS:
    (default)       Scan for duplicates and optionally move them
    preview-scan    Scan and generate a preview snapshot for the GUI

SCAN OPTIONS:
    <ROOT>                      Root directory to scan
    [TARGET]                    Target directory for moving duplicates
    --root=<PATH>               Root directory (alternative syntax)
    --target=<PATH>             Target directory (alternative syntax)

PREVIEW-SCAN OPTIONS:
    <ROOT>                      Root directory to scan
    --root=<PATH>               Root directory (alternative syntax)
    --output=<PATH>             Output snapshot path
    --thumbnail-root=<PATH>     Directory for thumbnail cache

COMMON OPTIONS:
    --no-thread                 Disable parallel processing
    --rename-to-guid            Rename files to GUID format
    --detect-low-resolution     Flag low-resolution images
    --enable-classification     Enable AI image classification
    --classify                  Alias for --enable-classification
    --enable-feature-detection  Enable feature-based detection for crops (slower)
    --feature-detection         Alias for --enable-feature-detection
    --prefer-ultrawide-aspect-ratios Prefer ultrawide (21:9 / 9:21) images as keepers
    --prefer-display-aspect-ratios   Deprecated alias for --prefer-ultrawide-aspect-ratios
    -h, --help                  Show this help message
    -V, --version               Show version information

EXAMPLES:
    camden ./photos
    camden ./photos ./duplicates --enable-classification
    camden --root=./images --target=./dupes --detect-low-resolution
    camden preview-scan ./photos --classify --thumbnail-root=./thumbs
    camden ./photos --enable-feature-detection
"#,
        env!("CAMDEN_VERSION_FULL")
    )
}

impl Error for CliError {}

fn default_extensions() -> Vec<String> {
    vec![
        "jpg".to_string(),
        "jpeg".to_string(),
        "png".to_string(),
        "gif".to_string(),
        "bmp".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scan_root_only() {
        let command = Command::from_iter(vec![String::from("./images")]).unwrap();
        match command {
            Command::Scan(config) => {
                assert_eq!(config.root, PathBuf::from("./images"));
                assert!(config.target.is_none());
                assert_eq!(config.threading, ThreadingMode::Parallel);
                assert_eq!(config.extensions, default_extensions());
                assert!(!config.rename_to_guid);
                assert!(!config.detect_low_resolution);
                assert!(!config.enable_classification);
                assert!(!config.enable_feature_detection);
            }
            _ => panic!("expected scan command"),
        }
    }

    #[test]
    fn parses_scan_flags() {
        let command = Command::from_iter(vec![
            String::from("--root=./images"),
            String::from("--target=./duplicates"),
            String::from("--no-thread"),
            String::from("--rename-to-guid"),
            String::from("--detect-low-resolution"),
            String::from("--enable-classification"),
        ])
        .unwrap();
        match command {
            Command::Scan(config) => {
                assert_eq!(config.root, PathBuf::from("./images"));
                assert_eq!(config.target, Some(PathBuf::from("./duplicates")));
                assert_eq!(config.threading, ThreadingMode::Sequential);
                assert!(config.rename_to_guid);
                assert!(config.detect_low_resolution);
                assert!(config.enable_classification);
            }
            _ => panic!("expected scan command"),
        }
    }

    #[test]
    fn parses_preview_with_flags() {
        let command = Command::from_iter(vec![
            String::from("preview-scan"),
            String::from("--root=./images"),
            String::from("--output=./cache/groups.json"),
            String::from("--thumbnail-root=./thumbs"),
            String::from("--no-thread"),
            String::from("--classify"),
        ])
        .unwrap();
        match command {
            Command::Preview(config) => {
                assert_eq!(config.root, PathBuf::from("./images"));
                assert_eq!(config.output, PathBuf::from("./cache/groups.json"));
                assert_eq!(config.thumbnail_root, Some(PathBuf::from("./thumbs")));
                assert_eq!(config.threading, ThreadingMode::Sequential);
                assert!(config.enable_classification);
            }
            _ => panic!("expected preview command"),
        }
    }

    #[test]
    fn preview_requires_root() {
        let result = Command::from_iter(vec![String::from("preview-scan")]);
        assert!(matches!(result, Err(CliError::MissingRoot)));
    }

    #[test]
    fn prefer_ultrawide_flag_sets_field() {
        let command = Command::from_iter(vec![
            String::from("./images"),
            String::from("--prefer-ultrawide-aspect-ratios"),
        ])
        .unwrap();
        match command {
            Command::Scan(config) => assert!(config.prefer_ultrawide_aspect_ratios),
            _ => panic!("expected scan command"),
        }
    }

    #[test]
    fn prefer_display_alias_still_parses() {
        // Legacy flag --prefer-display-aspect-ratios must remain accepted.
        let command = Command::from_iter(vec![
            String::from("./images"),
            String::from("--prefer-display-aspect-ratios"),
        ])
        .unwrap();
        match command {
            Command::Scan(config) => assert!(config.prefer_ultrawide_aspect_ratios),
            _ => panic!("expected scan command"),
        }
    }
}
