use cxx_build::CFG;
use std::path::PathBuf;

fn main() {
    let libtorch_path = match std::env::var_os("LIBTORCH").map(PathBuf::from) {
        None => {
            panic!("set env var LIBTORCH to the path to libtorch")
        },
        Some(p) => p,
    };

    let lib_path = std::env::var_os("LIBTORCH_LIB").map(PathBuf::from).unwrap_or(libtorch_path.join("lib"));
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath={}",
        lib_path.to_string_lossy()
    );
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");

    // use empty prefix: https://cxx.rs/build/cargo.html#header-include-paths
    CFG.include_prefix = "";

    let torch_inc_base = libtorch_path.join("include");
    let torchapi_inc_path = torch_inc_base.join("torch/csrc/api/include");

    cxx_build::bridge("src/aoti.rs")
        .include(torch_inc_base)
        .include(torchapi_inc_path)
        .file("csrc/aoti.cc")
        .std("c++20")
        .warnings(false)
        .compile("cxxbridge_aoti_package");

    println!("cargo:rerun-if-changed=src/aoti.rs");
    println!("cargo:rerun-if-changed=csrc/aoti.h");
    println!("cargo:rerun-if-changed=csrc/aoti.cc");
}
