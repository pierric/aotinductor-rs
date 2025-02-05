[![Build](https://github.com/pierric/aotinductor-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/pierric/aotinductor-rs/actions/workflows/rust.yml)
[![Latest](https://img.shields.io/crates/v/aotinductor.svg)](https://crates.io/crates/aotinductor)
[![Documentation](https://docs.rs/aotinductor/badge.svg)](https://docs.rs/aotinductor)

# aotinductor-rs

Rust bindings for pytorch [AOTInductor](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html).

## Build

This crate requires the libtorch in the same version as [tch-rs](https://github.com/LaurentMazare/tch-rs). You need to set the environment variable `LIBTORCH` to the path to folder of the library.

- If you are using pytorch, then the path is in the site-packages folder: `.../lib/python3.xx/site-packages/torch/`
- If you have downloaded a libtorch, then the path is where you unpackaged the library.

## Getting Started

```rust
use aotinductor::ModelPackage;
use tch::Tensor;

if let Some(model) = ModelPackage::new("path/to/some.pt2") {
    let inp1 = Tensor::rand([1, 2], (tch::Kind::Float, tch::Device::Cpu));
    let inp2 = Tensor::rand([1, 4], (tch::Kind::Float, tch::Device::Cpu));
    let out: std::vec::Vec<Tensor> = loader.run(&vec![inp1, inp2]);
};
```
