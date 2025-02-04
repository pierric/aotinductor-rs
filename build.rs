use std::path::PathBuf;
use cxx_build::CFG;

fn main() {
    if let Some(lib_path) = std::env::var_os("LIBTORCH_LIB") {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.to_string_lossy());
    }
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");

    // use empty prefix: https://cxx.rs/build/cargo.html#header-include-paths
    CFG.include_prefix = "";

    let inc_path = std::env::var_os("LIBTORCH_INCLUDE").map(PathBuf::from);
    if inc_path.is_none() {
        panic!("set env var LIBTORCH_INCLUDE to the path to include folder of torch");
    }
    let torch_inc_base = inc_path.unwrap();
    let torchapi_inc_path = torch_inc_base.join("torch/csrc/api/include");

    cxx_build::bridge("src/aoti.rs")  // returns a cc::Build
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
