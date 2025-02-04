#pragma once

#include "rust/cxx.h"
#include "torch/csrc/api/include/torch/torch.h"
#include "torch/csrc/inductor/aoti_package/model_package_loader.h"
#include <memory>

std::unique_ptr<torch::inductor::AOTIModelPackageLoader>
aoti_model_package_load(rust::Str path);

struct TensorUniquePtr;
struct TensorPtr;

rust::Vec<TensorUniquePtr>
aoti_model_package_run(torch::inductor::AOTIModelPackageLoader &loader,
                       const rust::Vec<TensorPtr> &inputs);
