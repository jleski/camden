use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
use std::fs::File as FsFile;
use std::hash::Hasher;
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use twox_hash::XxHash64;
use walkdir::WalkDir;

#[derive(Serialize)]
struct IdenticalFiles {
    checksum: u64,
    files: Vec<String>,
}

const USE_THREADING: bool = true;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <folder_path> [target_directory]", args[0]);
        std::process::exit(1);
    }
    let folder_path = &args[1];
    let target_directory = args.get(2).cloned();

    let image_extensions = vec!["jpg", "jpeg", "png", "gif", "bmp"];
    let total_files = WalkDir::new(folder_path).into_iter().count() as u64;
    let checksum_map: Arc<Mutex<HashMap<u64, Vec<PathBuf>>>> = Arc::new(Mutex::new(HashMap::new()));
    let progress_bar = ProgressBar::new(total_files);
    let progress_bar = Arc::new(progress_bar);

    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    if USE_THREADING == true {
        WalkDir::new(folder_path)
            .into_iter()
            .par_bridge()
            .for_each(|entry| {
                process_entry(&entry, &checksum_map, &progress_bar, &image_extensions)
            });
    } else {
        for entry in WalkDir::new(folder_path) {
            process_entry(&entry, &checksum_map, &progress_bar, &image_extensions);
        }
    }
    progress_bar.finish_with_message("Scan complete");

    let final_map = Arc::try_unwrap(checksum_map).unwrap().into_inner().unwrap();
    print_identical_files(&final_map);
    output_json(&final_map, "identical_files.json");

    if let Some(target_dir) = target_directory {
        let target_path = Path::new(&target_dir);
        if !target_path.exists() {
            fs::create_dir_all(target_path).expect("Failed to create target directory");
        }
        match move_duplicate_files(&final_map, target_path) {
            Ok(_) => println!("Duplicate files moved to {}", target_dir),
            Err(e) => eprintln!("Error moving duplicate files: {}", e),
        }
    }
}

fn process_entry(
    entry: &Result<walkdir::DirEntry, walkdir::Error>,
    checksum_map: &Arc<Mutex<HashMap<u64, Vec<PathBuf>>>>,
    progress_bar: &Arc<ProgressBar>,
    image_extensions: &[&str],
) {
    if let Ok(entry) = entry {
        let path = entry.path();
        if path.is_file() && has_image_extension(path, image_extensions) {
            if let Ok(checksum) = compute_checksum(path) {
                let mut map = checksum_map.lock().unwrap();
                map.entry(checksum).or_default().push(path.to_path_buf());
            }
        }
        progress_bar.inc(1);
        progress_bar.set_message(format!("Scanning: {}", path.display()));
    }
}

fn has_image_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| extensions.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn compute_checksum(path: &Path) -> std::io::Result<u64> {
    let mut file = File::open(path)?;
    let mut hasher = XxHash64::default();
    let mut buffer = [0; 8192];

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.write(&buffer[..count]);
    }

    Ok(hasher.finish())
}

fn output_json(checksum_map: &HashMap<u64, Vec<PathBuf>>, output_file: &str) {
    let identical_files: Vec<IdenticalFiles> = checksum_map
        .iter()
        .filter(|(_, files)| files.len() > 1)
        .map(|(checksum, files)| IdenticalFiles {
            checksum: *checksum,
            files: files
                .iter()
                .map(|f| f.to_string_lossy().into_owned())
                .collect(),
        })
        .collect();

    let json = serde_json::to_string_pretty(&identical_files).unwrap();
    let mut file = FsFile::create(output_file).unwrap();
    file.write_all(json.as_bytes()).unwrap();
    println!("JSON output written to {}", output_file);
}

fn print_identical_files(checksum_map: &HashMap<u64, Vec<PathBuf>>) {
    for (_, files) in checksum_map.iter().filter(|(_, files)| files.len() > 1) {
        println!("Identical files:");
        for file in files {
            println!("  {}", file.display());
        }
        println!();
    }
}

fn move_duplicate_files(
    checksum_map: &HashMap<u64, Vec<PathBuf>>,
    target_directory: &Path,
) -> std::io::Result<()> {
    let total_files = checksum_map
        .values()
        .filter(|files| files.len() > 1)
        .map(|files| files.len() - 1)
        .sum::<usize>();
    let progress_bar = ProgressBar::new(total_files as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    for (_, files) in checksum_map.iter().filter(|(_, files)| files.len() > 1) {
        for (index, file) in files.iter().enumerate().skip(1) {
            let file_name = file.file_name().unwrap();
            let new_path = target_directory.join(file_name);
            fs::rename(file, &new_path)?;
            progress_bar.inc(1);
            progress_bar.set_message(format!("Moving: {}", new_path.display()));
        }
    }

    progress_bar.finish_with_message("File moving complete");
    Ok(())
}
