#!/bin/bash -e

declare -A tch_to_torch_version=(["v0.19.0"]="2.6.0")

TCH_VER=$(cargo tree -e normal -p tch | head -n1 | cut -d " " -f 2)
TORCH_VER=${tch_to_torch_version[$TCH_VER]}

if [ -z "$TORCH_VER" ]; then
  exit 1
fi

URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2Bcpu.zip"
curl -kLs -o /tmp/libtorch.zip $URL
