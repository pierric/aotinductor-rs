#include "csrc/aoti.h"
#include "src/aoti.rs.h"

std::unique_ptr<torch::inductor::AOTIModelPackageLoader>
aoti_model_package_load(rust::Str path) {
  return std::make_unique<torch::inductor::AOTIModelPackageLoader>(
      std::string(path));
}

rust::Vec<TensorUniquePtr>
aoti_model_package_run(torch::inductor::AOTIModelPackageLoader &loader,
                       const rust::Vec<TensorPtr> &inputs) {
  std::vector<at::Tensor> inputs_vec;
  for (auto &tp : inputs) {
    inputs_vec.push_back(*tp.ptr);
  }

  std::vector<at::Tensor> results = loader.run(inputs_vec);

  rust::Vec<TensorUniquePtr> ret_vec;
  for (auto &t : results) {
    ret_vec.push_back(TensorUniquePtr(std::make_unique<at::Tensor>(t)));
  }
  return ret_vec;
}
