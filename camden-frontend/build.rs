use std::fs;
use std::path::Path;

fn main() {
    slint_build::compile("ui/main.slint").expect("Failed to compile Slint UI");

    // Version embedding from VERSION.toml
    let version_path = Path::new("../VERSION.toml");
    println!("cargo:rerun-if-changed=../VERSION.toml");

    if let Ok(content) = fs::read_to_string(version_path) {
        let major = parse_field(&content, "major").unwrap_or(0);
        let minor = parse_field(&content, "minor").unwrap_or(1);
        let patch = parse_field(&content, "patch").unwrap_or(0);
        let build = parse_field(&content, "build").unwrap_or(1);

        println!(
            "cargo:rustc-env=CAMDEN_VERSION={}.{}.{}",
            major, minor, patch
        );
        println!("cargo:rustc-env=CAMDEN_BUILD={}", build);
        println!(
            "cargo:rustc-env=CAMDEN_VERSION_FULL=v{}.{}.{} (build {})",
            major, minor, patch, build
        );
    } else {
        // Fallback if VERSION.toml doesn't exist
        println!("cargo:rustc-env=CAMDEN_VERSION=0.1.0");
        println!("cargo:rustc-env=CAMDEN_BUILD=0");
        println!("cargo:rustc-env=CAMDEN_VERSION_FULL=v0.1.0 (build 0)");
    }
}

fn parse_field(content: &str, field: &str) -> Option<u32> {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(field)
            && let Some(value) = trimmed.split('=').nth(1)
        {
            return value.trim().parse().ok();
        }
    }
    None
}
