use std::cell::RefCell;
pub use cxx::{UniquePtr, Exception};

#[cxx::bridge]
mod aoti {

    // rust::Vec supports only a rust type. There is no direct way to pass
    // Vec<tch::Tensor> through FFI boundary. So we pass the pointers instead.
    // And the pointers are wrapped in the following two structures.
    //
    // TensorPtr for the input tensors.
    // TensorUniquePtr for the output tensors that will be cloned at the end.

    struct TensorUniquePtr {
        ptr: UniquePtr<Tensor>
    }

    struct TensorPtr {
        ptr: *const Tensor
    }

    extern "Rust" {
    }

    unsafe extern "C++" {
        include!("csrc/aoti.h");

        #[namespace = "torch::inductor"]
        type AOTIModelPackageLoader;

        // torch::Tensor === at::Tensor
        #[namespace = "torch"]
        type Tensor;

        fn aoti_model_package_load(path: &str) -> Result<UniquePtr<AOTIModelPackageLoader>>;
        fn aoti_model_package_run(loader: Pin<&mut AOTIModelPackageLoader>, inputs: &Vec<TensorPtr>)
            -> Vec<TensorUniquePtr>;
    }
}

pub struct ModelPackage(RefCell<UniquePtr<aoti::AOTIModelPackageLoader>>);

impl ModelPackage {
    pub fn new(path: &str) -> Result<Self, Exception>  {
        let mp = aoti::aoti_model_package_load(path)?;
        Ok(Self(RefCell::new(mp)))
    }

    pub fn run(&self, inputs: &Vec<tch::Tensor>) -> Vec<tch::Tensor> {
        let inputs = inputs.iter().map(|t| {
            aoti::TensorPtr{ptr: t.as_ptr() as *const aoti::Tensor}
        }).collect();

        let mut outputs = aoti::aoti_model_package_run(self.0.borrow_mut().pin_mut(), &inputs);

        unsafe {
            outputs.iter_mut().map(|t| {
                tch::Tensor::clone_from_ptr(t.ptr.as_mut_ptr() as *mut torch_sys::C_tensor)
            }).collect()
        }
    }
}
