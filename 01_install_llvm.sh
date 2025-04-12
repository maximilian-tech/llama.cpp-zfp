#!/bin/bash

module purge
module load foss/2023b CMake Ninja

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


# Configurable variables
LLVM_VERSION="20.1.2"
ARCHIVE_URL="https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-${LLVM_VERSION}.tar.gz"
ARCHIVE_NAME="llvmorg-${LLVM_VERSION}.tar.gz"
EXTRACTED_DIR="llvm-project-llvmorg-${LLVM_VERSION}"

# Base directories under /dev/shmem
BASE_DIR="/dev/shm/llvm-build"
LLVM_PROJECT_DIR="${BASE_DIR}/${EXTRACTED_DIR}"
BOOTSTRAP_BUILD_DIR="${BASE_DIR}/build-bootstrap"
FINAL_BUILD_DIR="${BASE_DIR}/build-final"
BOOTSTRAP_INSTALL_DIR="${BASE_DIR}/llvm-bootstrap"
FINAL_INSTALL_DIR="${SCRIPT_DIR}/tools/llvm"

# Create base directories
mkdir -p "$BASE_DIR" "$BOOTSTRAP_BUILD_DIR" "$FINAL_BUILD_DIR" "$BOOTSTRAP_INSTALL_DIR" "$FINAL_INSTALL_DIR"

# -----------------------------------------
# Step 0: Download and extract LLVM source
# -----------------------------------------
if [ ! -d "$LLVM_PROJECT_DIR" ]; then
  echo "Downloading LLVM project tarball (version ${LLVM_VERSION})..."
  cd "$BASE_DIR"
  wget "$ARCHIVE_URL" -O "$ARCHIVE_NAME"

  echo "Extracting tarball..."
  tar -xzf "$ARCHIVE_NAME"
else
  echo "LLVM project source already present in ${LLVM_PROJECT_DIR}"
fi

# -----------------------------------------
# Step 1: Build bootstrap LLVM with GCC
# -----------------------------------------
echo "=== Building LLVM bootstrap with GCC ==="
cd "$BOOTSTRAP_BUILD_DIR"

cmake -G Ninja "${LLVM_PROJECT_DIR}/llvm" \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BOOTSTRAP_INSTALL_DIR" \
  --fresh

echo "Running Ninja build for bootstrap..."
ninja

echo "Installing bootstrap LLVM to ${BOOTSTRAP_INSTALL_DIR}..."
ninja install

# -----------------------------------------
# Step 2: Build final LLVM with bootstrapped Clang
# -----------------------------------------
echo "=== Building Final LLVM with bootstrapped Clang ==="
# Prepend the bootstrapped Clang to PATH so that CMake picks it up.
export PATH="$BOOTSTRAP_INSTALL_DIR/bin:$PATH"

cd "$FINAL_BUILD_DIR"

cmake -G Ninja "${LLVM_PROJECT_DIR}/llvm" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;compiler-rt;lld;openmp;lldb;flang" \
  -DLLVM_ENABLE_RUNTIMES="libunwind;libcxx;libcxxabi" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_INSTALL_PREFIX="$FINAL_INSTALL_DIR" \
  --fresh

echo "Running Ninja build for final LLVM toolchain..."
ninja

echo "Installing final LLVM toolchain to ${FINAL_INSTALL_DIR}..."
ninja install

echo "=== Bootstrapping Completed ==="
echo "Final LLVM toolchain installed in: ${FINAL_INSTALL_DIR}"
