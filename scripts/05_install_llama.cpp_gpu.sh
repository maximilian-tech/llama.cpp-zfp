#!/bin/bash

set -euxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ROOT_DIR=${SCRIPT_DIR}/../

cd ${ROOT_DIR}/llama.cpp

source ${ROOT_DIR}/source_env.cpp_gpu

cmake -B build_gpu \
  --install-prefix=${ROOT_DIR}/llama.cpp-gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DGGML_CUDA=ON \
  -DGGML_ZFP_ENABLE=OFF \
  -DGGML_ZFP_MODE="" \
  -DCMAKE_CUDA_COMPILER=nvcc \
  --fresh

make -C build_gpu -j 8 
make -C build_gpu install 
