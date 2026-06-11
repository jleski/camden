use camden_core::{
    ResolutionTier, ScanConfig, ScanSummary, ThreadingMode, move_paths, scan,
    write_classification_report,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use slint::{Image, ModelRc, SharedString, Timer, TimerMode, VecModel};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

slint::include_modules!();

thread_local! {
    static IMAGE_CACHE: RefCell<HashMap<PathBuf, Image>> = RefCell::new(HashMap::new());
}

#[derive(Clone)]
struct InternalFile {
    path: PathBuf,
    display_name: String,
    info: String,
    size_bytes: u64,
    sort_date: Option<String>,
    selected: bool,
    thumbnail: Option<PathBuf>,
    resolution_tier: ResolutionTier,
    moderation_tier: String,
    tags: String,
    #[allow(dead_code)]
    dimensions: (u32, u32),
    orientation: i32, // 0=Landscape, 1=Portrait, 2=Square
    is_keep_candidate: bool,
}

#[derive(Clone)]
struct InternalGroup {
    fingerprint: String,
    files: Vec<InternalFile>,
    total_size_bytes: u64,
    reclaimable_bytes: u64,
}

use std::time::Instant;

#[derive(Default, Clone)]
struct AppState {
    groups: Vec<InternalGroup>,
    all_photos: Vec<InternalFile>,     // All scanned photos for gallery
    gallery_photos: Vec<InternalFile>, // Filtered gallery view
    scanning: bool,
    last_scan_duration: Option<std::time::Duration>,
    progress_bar: Option<Arc<ProgressBar>>, // Current scan progress bar
    scan_phase: Option<Arc<Mutex<String>>>, // Current scan phase message
}

#[derive(Serialize, Deserialize, Clone)]
struct AppSettings {
    last_root_path: Option<String>,
    last_target_path: Option<String>,
    archive_path: Option<String>,
    cache_path: Option<String>,
    dark_mode: bool,
    show_tags: bool,
    compact_cards: bool,
    scan_find_duplicates: bool,
    scan_classify_images: bool,
    scan_detect_low_res: bool,
    scan_feature_detection: bool,
    scan_rename_to_guid: bool,
    scan_prefer_display_aspect_ratios: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            last_root_path: None,
            last_target_path: None,
            archive_path: None,
            cache_path: None,
            dark_mode: true,
            show_tags: true,
            compact_cards: false,
            scan_find_duplicates: true,
            scan_classify_images: false,
            scan_detect_low_res: true,
            scan_feature_detection: true,
            scan_rename_to_guid: false,
            scan_prefer_display_aspect_ratios: false,
        }
    }
}

fn settings_file_path() -> Option<PathBuf> {
    dirs::config_dir().map(|mut path| {
        path.push("Camden");
        fs::create_dir_all(&path).ok();
        path.push("settings.json");
        path
    })
}

fn load_settings() -> AppSettings {
    settings_file_path()
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default()
}

fn save_settings(settings: &AppSettings) {
    if let Some(path) = settings_file_path() {
        if let Ok(content) = serde_json::to_string_pretty(settings) {
            let _ = fs::write(path, content);
        }
    }
}

fn main() -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;

    // Load settings and apply to UI
    let settings = Arc::new(Mutex::new(load_settings()));
    {
        let settings = settings.lock().unwrap();
        ui.set_root_path(
            settings
                .last_root_path
                .clone()
                .unwrap_or_else(|| default_initial_root())
                .into(),
        );
        ui.set_target_path(
            settings
                .last_target_path
                .clone()
                .unwrap_or_else(|| default_target_path())
                .into(),
        );
        ui.set_dark_mode(settings.dark_mode);
        ui.set_settings_show_tags(settings.show_tags);
        ui.set_settings_compact_cards(settings.compact_cards);
        ui.set_find_duplicates(settings.scan_find_duplicates);
        ui.set_enable_classification(settings.scan_classify_images);
        ui.set_detect_low_resolution(settings.scan_detect_low_res);
        ui.set_enable_feature_detection(settings.scan_feature_detection);
        ui.set_rename_to_guid(settings.scan_rename_to_guid);
        ui.set_prefer_display_aspect_ratios(settings.scan_prefer_display_aspect_ratios);
        if let Some(archive) = &settings.archive_path {
            ui.set_settings_archive_path(archive.clone().into());
        }
        if let Some(cache) = &settings.cache_path {
            ui.set_settings_cache_path(cache.clone().into());
        }
    }
    ui.set_status_text("Select a root directory and press Scan.".into());

    let state = Arc::new(Mutex::new(AppState::default()));
    let ui_weak = ui.as_weak();

    // Setup progress polling timer - MUST keep timer alive by storing it
    let progress_timer = Timer::default();
    {
        let state_clone = Arc::clone(&state);
        let ui_weak_clone = ui_weak.clone();
        progress_timer.start(
            TimerMode::Repeated,
            std::time::Duration::from_millis(200),
            move || {
                if let Ok(state_guard) = state_clone.lock() {
                    if let Some(progress_bar) = &state_guard.progress_bar {
                        let pos = progress_bar.position();
                        let len = progress_bar.length().unwrap_or(0);

                        // Get current phase message
                        let phase_msg = state_guard
                            .scan_phase
                            .as_ref()
                            .and_then(|p| p.lock().ok())
                            .map(|s| s.clone())
                            .unwrap_or_else(|| "Scanning".to_string());

                        if let Some(ui) = ui_weak_clone.upgrade() {
                            // Update file counters
                            ui.set_files_scanned(pos as i32);
                            ui.set_files_total(len as i32);

                            // Update progress (0.0 to 1.0)
                            if len > 0 {
                                let progress = pos as f32 / len as f32;
                                ui.set_scan_progress(progress);
                            }

                            // Update status text with phase and progress
                            if len > 0 {
                                ui.set_status_text(
                                    format!("{}: {} / {}", phase_msg, pos, len).into(),
                                );
                            }
                        }
                    }
                }
            },
        );
    }

    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        let settings_clone = Arc::clone(&settings);
        ui.on_scan_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let root_text = ui.get_root_path().trim().to_string();
                if root_text.is_empty() {
                    ui.set_status_text("Please enter a root directory before scanning.".into());
                    return;
                }

                let root_path = PathBuf::from(&root_text);
                if !root_path.exists() {
                    ui.set_status_text("Root directory does not exist.".into());
                    return;
                }

                // Save scan settings before starting scan
                if let Ok(mut settings_mut) = settings_clone.lock() {
                    settings_mut.scan_find_duplicates = ui.get_find_duplicates();
                    settings_mut.scan_classify_images = ui.get_enable_classification();
                    settings_mut.scan_detect_low_res = ui.get_detect_low_resolution();
                    settings_mut.scan_feature_detection = ui.get_enable_feature_detection();
                    settings_mut.scan_rename_to_guid = ui.get_rename_to_guid();
                    settings_mut.scan_prefer_display_aspect_ratios =
                        ui.get_prefer_display_aspect_ratios();
                    settings_mut.last_root_path = Some(root_text.clone());
                    save_settings(&settings_mut);
                }

                if let Ok(mut state_mut) = state.lock() {
                    state_mut.scanning = true;
                    state_mut.last_scan_duration = None;
                }
                ui.set_scanning(true);
                ui.set_scan_progress(0.0);
                ui.set_files_scanned(0);
                ui.set_files_total(0);
                ui.set_status_text("Scanning…".into());

                let rename_to_guid = ui.get_rename_to_guid();
                let detect_low_resolution = ui.get_detect_low_resolution();
                let enable_classification = ui.get_enable_classification();
                let enable_feature_detection = ui.get_enable_feature_detection();
                let prefer_display_aspect_ratios = ui.get_prefer_display_aspect_ratios();
                let config = build_scan_config(
                    rename_to_guid,
                    detect_low_resolution,
                    enable_classification,
                    enable_feature_detection,
                    prefer_display_aspect_ratios,
                );

                let ui_weak = ui_weak.clone();
                let state = Arc::clone(&state);

                std::thread::spawn(move || perform_scan(root_path, config, state, ui_weak));
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        let settings_clone = Arc::clone(&settings);
        ui.on_browse_root(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                if let Some(ui) = ui_weak.upgrade() {
                    let root = folder.to_string_lossy().to_string();
                    ui.set_root_path(root.clone().into());
                    if ui.get_target_path().is_empty() {
                        let target = folder.join("duplicates");
                        ui.set_target_path(target.to_string_lossy().to_string().into());
                    }

                    // Save to settings
                    if let Ok(mut settings_mut) = settings_clone.lock() {
                        settings_mut.last_root_path = Some(root);
                        save_settings(&settings_mut);
                    }
                }
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        let settings_clone = Arc::clone(&settings);
        ui.on_browse_target(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                if let Some(ui) = ui_weak.upgrade() {
                    let target = folder.to_string_lossy().to_string();
                    ui.set_target_path(target.clone().into());

                    // Save to settings
                    if let Ok(mut settings_mut) = settings_clone.lock() {
                        settings_mut.last_target_path = Some(target);
                        save_settings(&settings_mut);
                    }
                }
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_file_toggled(move |group_index, file_index| {
            let group_index = group_index as usize;
            let file_index = file_index as usize;

            if let Some(ui) = ui_weak.upgrade() {
                if let Ok(mut state_mut) = state.lock() {
                    if let Some(group) = state_mut.groups.get_mut(group_index) {
                        if let Some(file) = group.files.get_mut(file_index) {
                            file.selected = !file.selected;
                        }
                    }
                    let snapshot = state_mut.clone();
                    drop(state_mut);
                    refresh_ui(&ui, &snapshot, None);
                }
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_move_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let target_text = ui.get_target_path().trim().to_string();
                if target_text.is_empty() {
                    ui.set_status_text("Please choose a target directory.".into());
                    return;
                }

                let target_path = PathBuf::from(&target_text);
                let selected: Vec<PathBuf> = state
                    .lock()
                    .map(|state| {
                        state
                            .groups
                            .iter()
                            .flat_map(|group| {
                                group
                                    .files
                                    .iter()
                                    .filter(|file| file.selected)
                                    .map(|file| file.path.clone())
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if selected.is_empty() {
                    ui.set_status_text("No files selected to move.".into());
                    return;
                }

                ui.set_status_text(format!("Moving {} files…", selected.len()).into());
                let ui_weak = ui_weak.clone();
                let state = Arc::clone(&state);

                std::thread::spawn(move || perform_move(selected, target_path, state, ui_weak));
            }
        });
    }

    // Gallery filter changed callback
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_gallery_filter_changed(move || {
            if let Some(ui) = ui_weak.upgrade() {
                if let Ok(mut state_mut) = state.lock() {
                    let filtered = apply_gallery_filters(
                        &state_mut.all_photos,
                        ui.get_filter_show_landscape(),
                        ui.get_filter_show_portrait(),
                        ui.get_filter_show_square(),
                        ui.get_filter_show_high_res(),
                        ui.get_filter_show_mobile_res(),
                        ui.get_filter_show_low_res(),
                        ui.get_filter_show_safe(),
                        ui.get_filter_show_sensitive(),
                        ui.get_filter_show_mature(),
                        ui.get_filter_show_restricted(),
                        &ui.get_filter_tag_search().to_string(),
                    );
                    state_mut.gallery_photos = filtered;
                    let snapshot = state_mut.clone();
                    drop(state_mut);
                    refresh_ui(&ui, &snapshot, None);
                }
            }
        });
    }

    // Select all best in duplicates
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_select_all_best(move || {
            if let Some(ui) = ui_weak.upgrade() {
                if let Ok(mut state_mut) = state.lock() {
                    ensure_largest_selected(&mut state_mut.groups);
                    let snapshot = state_mut.clone();
                    drop(state_mut);
                    refresh_ui(&ui, &snapshot, None);
                }
            }
        });
    }

    // Select best in specific group
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_select_best_in_group(move |group_idx| {
            if let Some(ui) = ui_weak.upgrade() {
                if let Ok(mut state_mut) = state.lock() {
                    if let Some(group) = state_mut.groups.get_mut(group_idx as usize) {
                        if !group.files.is_empty() {
                            let keep_index = find_keep_index(&group.files);
                            for (index, file) in group.files.iter_mut().enumerate() {
                                file.selected = index != keep_index;
                                file.is_keep_candidate = index == keep_index;
                            }
                            group.reclaimable_bytes = group
                                .files
                                .iter()
                                .filter(|f| f.selected)
                                .map(|f| f.size_bytes)
                                .sum();
                        }
                    }
                    let snapshot = state_mut.clone();
                    drop(state_mut);
                    refresh_ui(&ui, &snapshot, None);
                }
            }
        });
    }

    // Archive selected files (uses archive_path from state or settings)
    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_archive_selected(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let archive_path_str = ui.get_settings_archive_path().to_string();
                if archive_path_str.is_empty() {
                    ui.set_status_text(
                        "Please set an archive path in Settings before archiving.".into(),
                    );
                    return;
                }
                let archive_path = PathBuf::from(archive_path_str);

                let selected: Vec<PathBuf> = state
                    .lock()
                    .map(|state| {
                        state
                            .groups
                            .iter()
                            .flat_map(|group| &group.files)
                            .filter(|file| file.selected)
                            .map(|file| file.path.clone())
                            .collect()
                    })
                    .unwrap_or_default();

                if selected.is_empty() {
                    ui.set_status_text("No files selected to archive.".into());
                    return;
                }

                ui.set_status_text(format!("Archiving {} files…", selected.len()).into());
                let ui_weak = ui_weak.clone();
                let state = Arc::clone(&state);

                std::thread::spawn(move || perform_move(selected, archive_path, state, ui_weak));
            }
        });
    }

    // Settings callbacks
    {
        let ui_weak = ui_weak.clone();
        ui.on_browse_archive_path(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                if let Some(ui) = ui_weak.upgrade() {
                    let path = folder.to_string_lossy().to_string();
                    ui.set_settings_archive_path(path.into());
                }
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        ui.on_browse_cache_path(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                if let Some(ui) = ui_weak.upgrade() {
                    let path = folder.to_string_lossy().to_string();
                    ui.set_settings_cache_path(path.into());
                }
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        ui.on_clear_cache(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let cache_path = ui.get_settings_cache_path().to_string();
                if !cache_path.is_empty() {
                    match fs::remove_dir_all(&cache_path) {
                        Ok(_) => {
                            ui.set_status_text(format!("Cache cleared: {}", cache_path).into());
                            ui.set_settings_cached_thumbnails(0);
                            ui.set_settings_cache_size_mb(0.0);
                        }
                        Err(e) => {
                            ui.set_status_text(format!("Failed to clear cache: {}", e).into());
                        }
                    }
                }
            }
        });
    }

    // Gallery action callbacks (stubs for now)
    {
        let state = Arc::clone(&state);
        let ui_weak_photo = ui_weak.clone();
        ui.on_gallery_photo_clicked(move |idx| {
            if let Some(ui) = ui_weak_photo.upgrade() {
                if let Ok(state_mut) = state.lock() {
                    if let Some(photo) = state_mut.gallery_photos.get(idx as usize) {
                        let photo_data = file_to_photo_data(photo, idx, -1);
                        ui.set_preview_photo(photo_data);
                        ui.set_show_preview_modal(true);
                    }
                }
            }
        });

        ui.on_gallery_photo_toggle_selected(|_idx| {
            // TODO: Toggle photo selection in gallery
        });

        ui.on_gallery_export_selected(|| {
            // TODO: Export selected photos
        });

        ui.on_gallery_archive_selected(|| {
            // TODO: Archive selected photos from gallery
        });

        ui.on_gallery_select_all(|| {
            // TODO: Select all filtered photos
        });

        ui.on_gallery_deselect_all(|| {
            // TODO: Deselect all photos
        });

        let ui_weak_clone = ui_weak.clone();
        let settings_clone = Arc::clone(&settings);
        ui.on_save_settings(move || {
            if let Some(ui) = ui_weak_clone.upgrade() {
                if let Ok(mut settings_mut) = settings_clone.lock() {
                    settings_mut.dark_mode = ui.get_dark_mode();
                    settings_mut.show_tags = ui.get_settings_show_tags();
                    settings_mut.compact_cards = ui.get_settings_compact_cards();
                    let archive = ui.get_settings_archive_path().to_string();
                    if !archive.is_empty() {
                        settings_mut.archive_path = Some(archive);
                    }
                    let cache = ui.get_settings_cache_path().to_string();
                    if !cache.is_empty() {
                        settings_mut.cache_path = Some(cache);
                    }
                    save_settings(&settings_mut);
                    ui.set_status_text("Settings saved.".into());
                }
            }
        });

        let ui_weak_clone = ui_weak.clone();
        let settings_clone = Arc::clone(&settings);
        ui.on_reset_defaults(move || {
            if let Some(ui) = ui_weak_clone.upgrade() {
                let defaults = AppSettings::default();
                ui.set_dark_mode(defaults.dark_mode);
                ui.set_settings_show_tags(defaults.show_tags);
                ui.set_settings_compact_cards(defaults.compact_cards);
                if let Ok(mut settings_mut) = settings_clone.lock() {
                    *settings_mut = defaults;
                    save_settings(&settings_mut);
                }
                ui.set_status_text("Settings reset to defaults.".into());
            }
        });

        ui.on_archive_others_in_group(|_idx| {
            // TODO: Archive all files in group except keep candidate
        });

        // Preview modal callbacks
        let ui_weak_clone = ui_weak.clone();
        ui.on_preview_close(move || {
            if let Some(ui) = ui_weak_clone.upgrade() {
                ui.set_show_preview_modal(false);
            }
        });
    }

    ui.run()
}

fn perform_scan(
    root: PathBuf,
    config: ScanConfig,
    state: Arc<Mutex<AppState>>,
    ui_weak: slint::Weak<MainWindow>,
) {
    let start_time = Instant::now();
    let classification_enabled = config.enable_classification;

    // Count total entries to set progress bar length
    let total_entries = camden_core::count_entries(&root);

    // Create a progress bar that the UI can poll
    let progress_bar = Arc::new(ProgressBar::new(total_entries));
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    progress_bar.set_message("Scanning");

    // Create a phase string for communicating scan stage to UI
    let scan_phase = Arc::new(Mutex::new("Scanning files".to_string()));

    // Store progress bar and phase in state so UI can poll it
    if let Ok(mut state_mut) = state.lock() {
        state_mut.progress_bar = Some(Arc::clone(&progress_bar));
        state_mut.scan_phase = Some(Arc::clone(&scan_phase));
    }

    let summary = scan(&root, &config, &progress_bar, Some(&scan_phase));
    let groups = map_summary(&summary);
    let duration = start_time.elapsed();

    // Write classification report if classification was enabled
    if classification_enabled {
        if let Err(e) = write_classification_report(&summary, &root) {
            eprintln!("Failed to write classification report: {}", e);
        }
    }

    // Collect ALL photos for gallery (not just actionable groups)
    // This ensures the gallery shows all scanned photos, including unique high-res images
    let all_photos: Vec<InternalFile> = map_all_photos(&summary);

    if let Ok(mut state_mut) = state.lock() {
        state_mut.groups = groups;
        state_mut.all_photos = all_photos.clone();
        state_mut.gallery_photos = all_photos; // Initially show all photos
        state_mut.scanning = false;
        state_mut.last_scan_duration = Some(duration);
        state_mut.progress_bar = None; // Clear progress bar
        state_mut.scan_phase = None; // Clear scan phase
    }

    let duplicate_count = summary.duplicate_groups().count();
    let mobile_count = summary
        .groups
        .iter()
        .filter(|g| g.files.len() == 1 && g.files[0].resolution_tier == ResolutionTier::Mobile)
        .count();
    let low_res_count = summary
        .groups
        .iter()
        .filter(|g| g.files.len() == 1 && g.files[0].resolution_tier == ResolutionTier::Low)
        .count();

    // Count classified files
    let classified_count = if classification_enabled {
        summary
            .groups
            .iter()
            .flat_map(|g| &g.files)
            .filter(|f| f.moderation_tier.is_some())
            .count()
    } else {
        0
    };

    let status = if classification_enabled && classified_count > 0 {
        format!(
            "Scan complete in {:.2}s: {} duplicates, {} mobile-only, {} low-res, {} classified.",
            duration.as_secs_f64(),
            duplicate_count,
            mobile_count,
            low_res_count,
            classified_count
        )
    } else {
        format!(
            "Scan complete in {:.2}s: {} duplicates, {} mobile-only, {} low-res.",
            duration.as_secs_f64(),
            duplicate_count,
            mobile_count,
            low_res_count
        )
    };

    let state_clone = Arc::clone(&state);
    slint::invoke_from_event_loop(move || {
        if let Some(ui) = ui_weak.upgrade() {
            if let Ok(state_ref) = state_clone.lock() {
                ui.set_scanning(state_ref.scanning);
                refresh_ui(&ui, &state_ref, Some(status.clone()));
            }
        }
    })
    .ok();
}

fn perform_move(
    paths: Vec<PathBuf>,
    target: PathBuf,
    state: Arc<Mutex<AppState>>,
    ui_weak: slint::Weak<MainWindow>,
) {
    let create_target = fs::create_dir_all(&target);
    if let Err(err) = create_target {
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_status_text(format!("Failed to create target directory: {}", err).into());
            }
        })
        .ok();
        return;
    }

    let move_result = move_paths(&paths, &target);

    let status_text = match move_result {
        Ok(stats) => {
            let moved_set: HashSet<PathBuf> = paths.into_iter().collect();
            if let Ok(mut state_mut) = state.lock() {
                for group in state_mut.groups.iter_mut() {
                    group.files.retain(|file| !moved_set.contains(&file.path));
                }
                state_mut.groups.retain(|group| !group.files.is_empty());
                ensure_largest_selected(&mut state_mut.groups);
            }
            format!("Moved {} files to {}", stats.moved, target.display())
        }
        Err(err) => format!("Failed to move files: {}", err),
    };

    let state_clone = Arc::clone(&state);
    slint::invoke_from_event_loop(move || {
        if let Some(ui) = ui_weak.upgrade() {
            if let Ok(state_ref) = state_clone.lock() {
                ui.set_scanning(state_ref.scanning);
                refresh_ui(&ui, &state_ref, Some(status_text.clone()));
            }
        }
    })
    .ok();
}

fn refresh_ui(ui: &MainWindow, state: &AppState, status_override: Option<String>) {
    // Legacy compatibility - set old groups property
    let legacy_model = build_group_model(&state.groups);
    ui.set_groups(legacy_model);

    // New UI - set duplicate_groups and stats
    let duplicate_groups_model = build_duplicate_groups_model(&state.groups);
    ui.set_duplicate_groups(duplicate_groups_model);

    let duplicate_stats = calculate_duplicate_stats(&state.groups);
    ui.set_duplicate_stats(duplicate_stats);

    // Set gallery photos (filtered view)
    let gallery_model = build_gallery_photos_model(&state.gallery_photos);
    ui.set_gallery_photos(gallery_model);

    // Calculate and set gallery stats
    let gallery_stats = calculate_gallery_stats(&state.all_photos, &state.gallery_photos);
    ui.set_gallery_stats(gallery_stats);

    ui.set_scanning(state.scanning);
    if !state.scanning {
        ui.set_scan_progress(0.0);
    }

    let status = status_override.unwrap_or_else(|| format_status(state));
    ui.set_status_text(status.into());
}

fn build_group_model(groups: &[InternalGroup]) -> ModelRc<GroupData> {
    let group_data: Vec<GroupData> = groups
        .iter()
        .map(|group| {
            let files: Vec<FileData> = group
                .files
                .iter()
                .map(|file| FileData {
                    display_name: SharedString::from(file.display_name.clone()),
                    info: SharedString::from(file.info.clone()),
                    selected: file.selected,
                    thumbnail: file
                        .thumbnail
                        .as_ref()
                        .and_then(|path| load_thumbnail(path))
                        .unwrap_or_default(),
                    resolution_tier: resolution_tier_to_int(file.resolution_tier),
                    moderation_tier: SharedString::from(file.moderation_tier.clone()),
                    tags: SharedString::from(file.tags.clone()),
                })
                .collect();
            GroupData {
                fingerprint: SharedString::from(group.fingerprint.clone()),
                file_count: group.files.len() as i32,
                files: ModelRc::from(Rc::new(VecModel::from(files))),
            }
        })
        .collect();

    ModelRc::from(Rc::new(VecModel::from(group_data)))
}

/// Map all photos from summary for gallery view (includes all scanned photos)
fn map_all_photos(summary: &ScanSummary) -> Vec<InternalFile> {
    summary
        .groups
        .iter()
        .flat_map(|group| {
            group.files.iter().map(|file| {
                let display_name = file
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let info = format_file_info(file);
                let sort_date = file.captured_at.clone().or_else(|| file.modified.clone());
                let dimensions = (file.dimensions.0 as u32, file.dimensions.1 as u32);
                let orientation = classify_orientation(dimensions.0, dimensions.1);

                InternalFile {
                    path: file.path.clone(),
                    display_name,
                    info,
                    size_bytes: file.size_bytes,
                    sort_date,
                    selected: false,
                    thumbnail: file.thumbnail.clone(),
                    resolution_tier: file.resolution_tier,
                    moderation_tier: file.moderation_tier.clone().unwrap_or_default(),
                    tags: file.tags.join(", "),
                    dimensions,
                    orientation,
                    is_keep_candidate: false,
                }
            })
        })
        .collect()
}

fn map_summary(summary: &ScanSummary) -> Vec<InternalGroup> {
    summary
        .actionable_groups()
        .map(|group| {
            let fingerprint = format!("{:016x}", group.fingerprint);
            let mut files: Vec<InternalFile> = group
                .files
                .iter()
                .map(|file| {
                    let display_name = file
                        .path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let info = format_file_info(file);
                    let sort_date = file.captured_at.clone().or_else(|| file.modified.clone());
                    let dimensions = (file.dimensions.0 as u32, file.dimensions.1 as u32);
                    let orientation = classify_orientation(dimensions.0, dimensions.1);

                    InternalFile {
                        path: file.path.clone(),
                        display_name,
                        info,
                        size_bytes: file.size_bytes,
                        sort_date,
                        selected: false,
                        thumbnail: file.thumbnail.clone(),
                        resolution_tier: file.resolution_tier,
                        moderation_tier: file.moderation_tier.clone().unwrap_or_default(),
                        tags: file.tags.join(", "),
                        dimensions,
                        orientation,
                        is_keep_candidate: false,
                    }
                })
                .collect();

            // For duplicate groups: select all except the best one to keep
            // For resolution singletons: pre-select only if Low tier
            if files.len() > 1 {
                let keep_index = find_keep_index(&files);
                for (index, file) in files.iter_mut().enumerate() {
                    file.selected = index != keep_index;
                    file.is_keep_candidate = index == keep_index;
                }
            } else if files.len() == 1 && files[0].resolution_tier.should_preselect() {
                files[0].selected = true;
            }

            // Calculate group totals
            let total_size_bytes: u64 = files.iter().map(|f| f.size_bytes).sum();
            let reclaimable_bytes: u64 = files
                .iter()
                .filter(|f| f.selected)
                .map(|f| f.size_bytes)
                .sum();

            InternalGroup {
                fingerprint,
                files,
                total_size_bytes,
                reclaimable_bytes,
            }
        })
        .collect()
}

fn find_keep_index(files: &[InternalFile]) -> usize {
    if files.is_empty() {
        return 0;
    }

    let max_size = files.iter().map(|f| f.size_bytes).max().unwrap_or(0);

    files
        .iter()
        .enumerate()
        .filter(|(_, f)| f.size_bytes == max_size)
        .max_by(|(_, a), (_, b)| {
            let date_cmp = a.sort_date.cmp(&b.sort_date);
            if date_cmp != std::cmp::Ordering::Equal {
                return date_cmp;
            }
            b.path.cmp(&a.path)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_file(name: &str, size: u64, date: Option<&str>) -> InternalFile {
        InternalFile {
            path: PathBuf::from(name),
            display_name: name.to_string(),
            info: String::new(),
            size_bytes: size,
            sort_date: date.map(|s| s.to_string()),
            selected: false,
            thumbnail: None,
            resolution_tier: ResolutionTier::High,
            moderation_tier: String::new(),
            tags: String::new(),
            dimensions: (1920, 1080),
            orientation: 0,
            is_keep_candidate: false,
        }
    }

    #[test]
    fn keep_largest() {
        let files = vec![
            make_file("small.jpg", 100, Some("2023-01-01")),
            make_file("large.jpg", 200, Some("2023-01-01")),
        ];
        assert_eq!(find_keep_index(&files), 1);
    }

    #[test]
    fn keep_newest_when_size_same() {
        let files = vec![
            make_file("old.jpg", 100, Some("2023-01-01")),
            make_file("new.jpg", 100, Some("2023-01-02")),
        ];
        assert_eq!(find_keep_index(&files), 1);
    }

    #[test]
    fn keep_alphabetical_when_size_and_date_same() {
        // "b.jpg" should be kept because we use b.path.cmp(&a.path) which is reverse lexical sort?
        // Wait, the logic is `b.path.cmp(&a.path)`.
        // If a="a.jpg", b="b.jpg".
        // a.cmp(b) is Less. b.cmp(a) is Greater.
        // max_by will pick the one that is Greater.
        // So "b.jpg" > "a.jpg" in string comparison? Yes.
        // So b.path.cmp(&a.path):
        // compare(current_max, candidate).
        // if candidate > current_max, candidate becomes max.
        // Let's trace.
        // Logic: .max_by(|(_, a), (_, b)| ... b.path.cmp(&a.path))
        // This is tricky. max_by returns the element that yields Ordering::Greater when compared to others.
        // The closure compares `a` (left) and `b` (right).
        // If we want "a" to be "greater" (selected) than "b", we should return Greater.
        // We want stable sort order? Usually "path" tiebreaker is purely deterministic.
        // Let's assume we want "first" file alphabetically if all else equal?
        // If we want "a.jpg" to be kept over "b.jpg", "a" should be "better".
        // If b.path.cmp(&a.path) is used:
        // "a.jpg" vs "b.jpg" -> "b".cmp("a") -> Greater. So "a" is Greater than "b"? No.
        // max_by(cmp): if cmp(a, b) == Greater, a is max.
        // cmp("a", "b") -> "b".cmp("a") -> Greater.
        // So "a" is considered "greater" (better) than "b".
        // So "a.jpg" should be kept.
        // Let's test this assumption.

        let files = vec![
            make_file("a.jpg", 100, Some("2023-01-01")),
            make_file("b.jpg", 100, Some("2023-01-01")),
        ];
        assert_eq!(find_keep_index(&files), 0); // a.jpg
    }

    #[test]
    fn keep_largest_ignores_date() {
        let files = vec![
            make_file("small_new.jpg", 100, Some("2023-01-02")),
            make_file("large_old.jpg", 200, Some("2023-01-01")),
        ];
        assert_eq!(find_keep_index(&files), 1);
    }

    #[test]
    fn handle_missing_dates() {
        let files = vec![
            make_file("no_date.jpg", 100, None),
            make_file("with_date.jpg", 100, Some("2023-01-01")),
        ];
        // Some("...") > None is true for Option.
        // So with_date should be kept.
        assert_eq!(find_keep_index(&files), 1);
    }
}

fn ensure_largest_selected(groups: &mut [InternalGroup]) {
    for group in groups.iter_mut() {
        if group.files.is_empty() {
            continue;
        }
        let keep_index = find_keep_index(&group.files);
        for (index, file) in group.files.iter_mut().enumerate() {
            file.selected = index != keep_index;
            file.is_keep_candidate = index == keep_index;
        }

        // Recalculate group totals
        group.total_size_bytes = group.files.iter().map(|f| f.size_bytes).sum();
        group.reclaimable_bytes = group
            .files
            .iter()
            .filter(|f| f.selected)
            .map(|f| f.size_bytes)
            .sum();
    }
}

fn format_file_info(file: &camden_core::DuplicateEntry) -> String {
    let size = if file.size_bytes == 0 {
        "0 B".to_string()
    } else {
        format_size(file.size_bytes)
    };
    let dimensions = format!("{}×{}", file.dimensions.0, file.dimensions.1);
    let timestamp = file
        .captured_at
        .as_deref()
        .or_else(|| file.modified.as_deref())
        .unwrap_or("unknown");
    format!("{} • {} • {}", size, dimensions, timestamp)
}

fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit = 0usize;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{:.1} {}", size, UNITS[unit])
    }
}

fn classify_orientation(width: u32, height: u32) -> i32 {
    if width > height {
        0 // Landscape
    } else if height > width {
        1 // Portrait
    } else {
        2 // Square
    }
}

fn resolution_tier_to_int(tier: ResolutionTier) -> i32 {
    match tier {
        ResolutionTier::High => 0,
        ResolutionTier::Mobile => 1,
        ResolutionTier::Low => 2,
    }
}

// Convert InternalFile to PhotoData for new UI
fn calculate_aspect_ratio(width: u32, height: u32) -> String {
    if width == 0 || height == 0 {
        return String::new();
    }

    let ratio = width as f32 / height as f32;

    const RATIOS: &[(f32, &str)] = &[
        // Landscape
        (1.0, "1:1"),
        (5.0 / 4.0, "5:4"),     // 1.25
        (4.0 / 3.0, "4:3"),     // 1.333
        (3.0 / 2.0, "3:2"),     // 1.5
        (16.0 / 10.0, "16:10"), // 1.6
        (5.0 / 3.0, "5:3"),     // 1.666
        (16.0 / 9.0, "16:9"),   // 1.777
        (21.0 / 9.0, "21:9"),   // 2.333
        (2.39, "2.39:1"),       // 2.39, anamorphic
        // Portrait
        (4.0 / 5.0, "4:5"),
        (3.0 / 4.0, "3:4"),
        (2.0 / 3.0, "2:3"),
        (10.0 / 16.0, "10:16"),
        (3.0 / 5.0, "3:5"),
        (9.0 / 16.0, "9:16"),
        (9.0 / 21.0, "9:21"),
    ];

    let mut closest_ratio = "";
    let mut min_diff = f32::MAX;

    for &(r, name) in RATIOS {
        let diff = (ratio - r).abs();
        if diff < min_diff {
            min_diff = diff;
            closest_ratio = name;
        }
    }

    // Threshold to decide if we use the common ratio or the exact one.
    // If the difference is less than 2%, use the common ratio.
    if min_diff < 0.02 {
        return closest_ratio.to_string();
    }

    // Calculate GCD to reduce fraction for non-common ratios
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    let divisor = gcd(width, height);
    let w = width / divisor;
    let h = height / divisor;
    format!("{}:{}", w, h)
}

fn file_to_photo_data(file: &InternalFile, id: i32, group_id: i32) -> PhotoData {
    let aspect_ratio = calculate_aspect_ratio(file.dimensions.0, file.dimensions.1);
    PhotoData {
        id,
        display_name: SharedString::from(file.display_name.clone()),
        info: SharedString::from(file.info.clone()),
        thumbnail: file
            .thumbnail
            .as_ref()
            .and_then(|path| load_thumbnail(path))
            .unwrap_or_default(),
        selected: file.selected,
        resolution_tier: resolution_tier_to_int(file.resolution_tier),
        orientation: file.orientation,
        moderation_tier: SharedString::from(file.moderation_tier.clone()),
        tags: SharedString::from(file.tags.clone()),
        aspect_ratio: SharedString::from(aspect_ratio),
        is_keep_candidate: file.is_keep_candidate,
        group_id,
    }
}

// Build DuplicateGroup models for new UI
fn build_duplicate_groups_model(groups: &[InternalGroup]) -> ModelRc<DuplicateGroup> {
    let mut photo_id = 0i32;
    let duplicate_groups: Vec<DuplicateGroup> = groups
        .iter()
        .enumerate()
        .map(|(group_idx, group)| {
            let files: Vec<PhotoData> = group
                .files
                .iter()
                .map(|file| {
                    let photo = file_to_photo_data(file, photo_id, group_idx as i32);
                    photo_id += 1;
                    photo
                })
                .collect();

            DuplicateGroup {
                fingerprint: SharedString::from(group.fingerprint.clone()),
                files: ModelRc::from(Rc::new(VecModel::from(files))),
                total_size_bytes: group.total_size_bytes as i32,
                reclaimable_bytes: group.reclaimable_bytes as i32,
            }
        })
        .collect();

    ModelRc::from(Rc::new(VecModel::from(duplicate_groups)))
}

// Calculate DuplicateStats from groups
fn calculate_duplicate_stats(groups: &[InternalGroup]) -> DuplicateStats {
    let total_groups = groups.len() as i32;
    let total_files: i32 = groups.iter().map(|g| g.files.len() as i32).sum();
    let total_duplicates = groups
        .iter()
        .filter(|g| g.files.len() > 1)
        .map(|g| (g.files.len() - 1) as i32)
        .sum();
    let reclaimable_bytes: u64 = groups.iter().map(|g| g.reclaimable_bytes).sum();
    let reclaimable_mb = (reclaimable_bytes as f64) / (1024.0 * 1024.0);
    let selected_count: i32 = groups
        .iter()
        .flat_map(|g| &g.files)
        .filter(|f| f.selected)
        .count() as i32;

    DuplicateStats {
        total_groups,
        total_files,
        total_duplicates,
        reclaimable_mb: reclaimable_mb as f32,
        selected_count,
    }
}

// Build gallery photos model
fn build_gallery_photos_model(photos: &[InternalFile]) -> ModelRc<PhotoData> {
    let photo_data: Vec<PhotoData> = photos
        .iter()
        .enumerate()
        .map(|(idx, file)| file_to_photo_data(file, idx as i32, -1))
        .collect();

    ModelRc::from(Rc::new(VecModel::from(photo_data)))
}

// Calculate GalleryStats
fn calculate_gallery_stats(
    all_photos: &[InternalFile],
    filtered_photos: &[InternalFile],
) -> GalleryStats {
    let total_photos = all_photos.len() as i32;
    let filtered_photos_count = filtered_photos.len() as i32;
    let selected_count = filtered_photos.iter().filter(|f| f.selected).count() as i32;

    let landscape_count = all_photos.iter().filter(|f| f.orientation == 0).count() as i32;
    let portrait_count = all_photos.iter().filter(|f| f.orientation == 1).count() as i32;
    let square_count = all_photos.iter().filter(|f| f.orientation == 2).count() as i32;

    GalleryStats {
        total_photos,
        filtered_photos: filtered_photos_count,
        selected_count,
        landscape_count,
        portrait_count,
        square_count,
    }
}

// Apply gallery filters
fn apply_gallery_filters(
    photos: &[InternalFile],
    show_landscape: bool,
    show_portrait: bool,
    show_square: bool,
    show_high_res: bool,
    show_mobile_res: bool,
    show_low_res: bool,
    show_safe: bool,
    show_sensitive: bool,
    show_mature: bool,
    show_restricted: bool,
    tag_search: &str,
) -> Vec<InternalFile> {
    photos
        .iter()
        .filter(|p| match p.orientation {
            0 => show_landscape,
            1 => show_portrait,
            2 => show_square,
            _ => true,
        })
        .filter(|p| match p.resolution_tier {
            ResolutionTier::High => show_high_res,
            ResolutionTier::Mobile => show_mobile_res,
            ResolutionTier::Low => show_low_res,
        })
        .filter(|p| {
            let tier = p.moderation_tier.as_str();
            match tier {
                "" | "Safe" => show_safe,
                "Sensitive" => show_sensitive,
                "Mature" => show_mature,
                "Restricted" => show_restricted,
                _ => true,
            }
        })
        .filter(|p| {
            tag_search.is_empty() || p.tags.to_lowercase().contains(&tag_search.to_lowercase())
        })
        .cloned()
        .collect()
}

fn format_status(state: &AppState) -> String {
    if state.scanning {
        return "Scanning…".to_string();
    }
    let group_count = state.groups.len();
    let mut file_count = 0usize;
    let mut selected_count = 0usize;
    for group in &state.groups {
        file_count += group.files.len();
        selected_count += group.files.iter().filter(|file| file.selected).count();
    }

    let time_info = if let Some(d) = state.last_scan_duration {
        format!(" • Time: {:.2}s", d.as_secs_f64())
    } else {
        String::new()
    };

    format!(
        "Groups: {} • Files: {} • Selected: {}{}",
        group_count, file_count, selected_count, time_info
    )
}

fn build_scan_config(
    rename_to_guid: bool,
    detect_low_resolution: bool,
    enable_classification: bool,
    enable_feature_detection: bool,
    prefer_display_aspect_ratios: bool,
) -> ScanConfig {
    let mut config = ScanConfig::new(default_extensions(), ThreadingMode::Parallel);
    if let Some(mut dir) = dirs::data_local_dir() {
        dir.push("Camden");
        dir.push("thumbnails");
        config = config.with_thumbnail_root(dir);
    }
    config
        .with_guid_rename(rename_to_guid)
        .with_low_resolution_detection(detect_low_resolution)
        .with_classification(enable_classification)
        .with_feature_detection(enable_feature_detection)
        .with_prefer_display_aspect_ratios(prefer_display_aspect_ratios)
}

fn default_extensions() -> Vec<String> {
    ["jpg", "jpeg", "png", "gif", "bmp", "webp"]
        .iter()
        .map(|ext| ext.to_string())
        .collect()
}

fn default_initial_root() -> String {
    dirs::picture_dir()
        .or_else(|| dirs::home_dir())
        .unwrap_or_else(|| PathBuf::from("C:\\"))
        .to_string_lossy()
        .to_string()
}

fn default_target_path() -> String {
    dirs::data_local_dir()
        .map(|mut dir| {
            dir.push("Camden");
            dir.push("duplicates");
            dir.to_string_lossy().to_string()
        })
        .unwrap_or_else(|| String::from("C:\\Camden\\duplicates"))
}

fn load_thumbnail(path: &Path) -> Option<Image> {
    IMAGE_CACHE.with(|cache_cell| {
        {
            let cache = cache_cell.borrow();
            if let Some(image) = cache.get(path) {
                return Some(image.clone());
            }
        }

        let mut cache = cache_cell.borrow_mut();
        if let Ok(image) = Image::load_from_path(path) {
            cache.insert(path.to_path_buf(), image.clone());
            return Some(image);
        }

        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("webp"))
            .unwrap_or(false)
        {
            let png_path = path.with_extension("png");
            if let Some(image) = cache.get(&png_path) {
                let image = image.clone();
                cache.insert(path.to_path_buf(), image.clone());
                return Some(image);
            }
            if let Ok(image) = Image::load_from_path(&png_path) {
                cache.insert(png_path.clone(), image.clone());
                cache.insert(path.to_path_buf(), image.clone());
                return Some(image);
            }
        }

        None
    })
}
