use crate::progress;
use crate::scanner::ScanSummary;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveStats {
    pub moved: usize,
}

#[derive(Debug)]
pub enum MoveError {
    Io {
        source: std::io::Error,
        path: PathBuf,
    },
    MissingFileName(PathBuf),
}

impl Display for MoveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { source, path } => write!(f, "failed to move {}: {}", path.display(), source),
            Self::MissingFileName(path) => write!(f, "file name not found for {}", path.display()),
        }
    }
}

impl Error for MoveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub fn move_duplicates(
    summary: &ScanSummary,
    target_directory: &Path,
) -> Result<MoveStats, MoveError> {
    let total = total_duplicates(summary) as u64;
    let progress_bar = ProgressBar::new(total);
    progress_bar.set_style(progress::default_style());

    let mut moved = 0;
    for group in summary.groups.iter().filter(|group| group.files.len() > 1) {
        for entry in group.files.iter().skip(1) {
            let source = &entry.path;
            let destination = resolve_destination(target_directory, source)?;
            fs::rename(source, &destination).map_err(|error| MoveError::Io {
                source: error,
                path: source.clone(),
            })?;
            moved += 1;
            progress_bar.inc(1);
            progress_bar.set_message(format!("Moving: {}", destination.display()));
        }
    }

    progress_bar.finish_with_message("File moving complete");
    Ok(MoveStats { moved })
}

pub fn move_paths(paths: &[PathBuf], target_directory: &Path) -> Result<MoveStats, MoveError> {
    fs::create_dir_all(target_directory).map_err(|source| MoveError::Io {
        source,
        path: target_directory.to_path_buf(),
    })?;

    let mut moved = 0;
    for source in paths {
        let destination = resolve_destination(target_directory, source)?;
        fs::rename(source, &destination).map_err(|error| MoveError::Io {
            source: error,
            path: source.clone(),
        })?;
        moved += 1;
    }
    Ok(MoveStats { moved })
}

/// Copies the given files into `target_directory`, keeping the originals in place.
///
/// Destination file names are de-duplicated the same way as [`move_paths`].
pub fn copy_paths(paths: &[PathBuf], target_directory: &Path) -> Result<MoveStats, MoveError> {
    fs::create_dir_all(target_directory).map_err(|source| MoveError::Io {
        source,
        path: target_directory.to_path_buf(),
    })?;

    let mut moved = 0;
    for source in paths {
        let destination = resolve_destination(target_directory, source)?;
        fs::copy(source, &destination).map_err(|error| MoveError::Io {
            source: error,
            path: source.clone(),
        })?;
        moved += 1;
    }
    Ok(MoveStats { moved })
}

/// Permanently deletes the given files from disk.
///
/// Stops and returns an error on the first failure; files deleted before the
/// failing entry remain deleted (the operation is not transactional).
pub fn delete_paths(paths: &[PathBuf]) -> Result<MoveStats, MoveError> {
    let mut moved = 0;
    for path in paths {
        fs::remove_file(path).map_err(|source| MoveError::Io {
            source,
            path: path.clone(),
        })?;
        moved += 1;
    }
    Ok(MoveStats { moved })
}

fn total_duplicates(summary: &ScanSummary) -> usize {
    summary
        .duplicate_groups()
        .map(|group| group.files.len() - 1)
        .sum()
}

fn resolve_destination(target_directory: &Path, source: &Path) -> Result<PathBuf, MoveError> {
    let file_name = source
        .file_name()
        .ok_or_else(|| MoveError::MissingFileName(source.to_path_buf()))?;

    let mut candidate = target_directory.join(file_name);
    if !candidate.exists() {
        return Ok(candidate);
    }

    let stem = source
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| stem.to_string())
        .unwrap_or_else(|| String::from("file"));
    let extension = source.extension().and_then(|ext| ext.to_str());
    let mut index = 1;

    loop {
        let mut name = format!("{} ({})", stem, index);
        if let Some(ext) = extension {
            name.push('.');
            name.push_str(ext);
        }
        candidate = target_directory.join(name);
        if !candidate.exists() {
            return Ok(candidate);
        }
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::{DuplicateEntry, DuplicateGroup, ScanSummary};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn move_duplicates_moves_files_and_generates_unique_names() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let keep = source_dir.path().join("dup.jpg");
        fs::write(&keep, b"same").unwrap();
        let nested_dir = source_dir.path().join("nested");
        fs::create_dir_all(&nested_dir).unwrap();
        let nested = nested_dir.join("dup.jpg");
        fs::write(&nested, b"same").unwrap();
        let other_dir = source_dir.path().join("other");
        fs::create_dir_all(&other_dir).unwrap();
        let other = other_dir.join("dup.jpg");
        fs::write(&other, b"same").unwrap();
        let existing = target_dir.path().join("dup.jpg");
        fs::write(&existing, b"existing").unwrap();
        let summary = ScanSummary {
            groups: vec![DuplicateGroup {
                fingerprint: 1,
                files: vec![
                    sample_entry(keep.clone()),
                    sample_entry(nested.clone()),
                    sample_entry(other.clone()),
                ],
            }],
        };
        let stats = move_duplicates(&summary, target_dir.path()).unwrap();
        assert_eq!(stats.moved, 2);
        assert!(keep.exists());
        assert!(!nested.exists());
        assert!(!other.exists());
        let mut names: Vec<_> = fs::read_dir(target_dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                String::from("dup (1).jpg"),
                String::from("dup (2).jpg"),
                String::from("dup.jpg")
            ]
        );
    }

    #[test]
    fn move_paths_transfers_files() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let first = source_dir.path().join("first.jpg");
        let second = source_dir.path().join("second.jpg");
        fs::write(&first, b"first").unwrap();
        fs::write(&second, b"second").unwrap();

        let stats = move_paths(&[first.clone(), second.clone()], target_dir.path()).unwrap();
        assert_eq!(stats.moved, 2);
        assert!(!first.exists());
        assert!(!second.exists());
        assert!(target_dir.path().join("first.jpg").exists());
        assert!(target_dir.path().join("second.jpg").exists());
    }

    #[test]
    fn copy_paths_keeps_originals() {
        let source_dir = tempdir().unwrap();
        let target_dir = tempdir().unwrap();
        let first = source_dir.path().join("first.jpg");
        fs::write(&first, b"first").unwrap();

        let stats = copy_paths(std::slice::from_ref(&first), target_dir.path()).unwrap();
        assert_eq!(stats.moved, 1);
        assert!(first.exists(), "original should remain after copy");
        assert!(target_dir.path().join("first.jpg").exists());
        assert_eq!(
            fs::read(target_dir.path().join("first.jpg")).unwrap(),
            b"first"
        );
    }

    #[test]
    fn delete_paths_removes_files() {
        let source_dir = tempdir().unwrap();
        let first = source_dir.path().join("first.jpg");
        let second = source_dir.path().join("second.jpg");
        fs::write(&first, b"first").unwrap();
        fs::write(&second, b"second").unwrap();

        let stats = delete_paths(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(stats.moved, 2);
        assert!(!first.exists());
        assert!(!second.exists());
    }

    #[test]
    fn delete_paths_errors_on_missing_file() {
        let source_dir = tempdir().unwrap();
        let missing = source_dir.path().join("missing.jpg");
        let result = delete_paths(&[missing]);
        assert!(result.is_err());
    }

    fn sample_entry(path: PathBuf) -> DuplicateEntry {
        DuplicateEntry {
            path,
            size_bytes: 4,
            dimensions: (1, 1),
            modified: None,
            captured_at: None,
            dominant_color: [0, 0, 0],
            confidence: 1.0,
            thumbnail: None,
            resolution_tier: crate::resolution::ResolutionTier::High,
            moderation_tier: None,
            tags: Vec::new(),
        }
    }
}
