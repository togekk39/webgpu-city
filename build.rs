use std::{env, fs};

fn dotenv_value(key: &str) -> Option<String> {
    let contents = fs::read_to_string(".env").ok()?;
    contents.lines().find_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (name, value) = line.split_once('=')?;
        if name.trim() == key {
            Some(value.trim().trim_matches(['\"', '\'']).to_owned())
        } else {
            None
        }
    })
}

fn main() {
    println!("cargo:rerun-if-env-changed=WEBGPU_CITY_GLTF_URL");
    println!("cargo:rerun-if-changed=.env");

    if env::var("WEBGPU_CITY_GLTF_URL").is_err() {
        if let Some(value) = dotenv_value("WEBGPU_CITY_GLTF_URL") {
            println!("cargo:rustc-env=WEBGPU_CITY_GLTF_URL={value}");
        }
    }
}
