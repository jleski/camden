use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::hash::Hasher;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use twox_hash::XxHash64;
use walkdir::WalkDir;

const USE_THREADING: bool = true;

fn main() {
    let args: Vec<String> = env::args().collect();
    let folder_path = if args.len() > 1 {
        &args[1]
    } else {
        eprintln!("Usage: {} <folder_path>", args[0]);
        std::process::exit(1);
    };

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
    print_identical_files(final_map);
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

/*fn compute_checksum(path: &Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0; 4096];

    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}*/

fn print_identical_files(checksum_map: HashMap<u64, Vec<PathBuf>>) {
    for (_, files) in checksum_map.iter().filter(|(_, files)| files.len() > 1) {
        println!("Identical files:");
        for file in files {
            println!("  {}", file.display());
        }
        println!();
    }
}
