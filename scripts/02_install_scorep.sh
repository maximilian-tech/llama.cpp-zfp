#!/bin/bash
set -euxo pipefail

# Determine the directory where the script resides
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=${SCRIPT_DIR}/..
TOOLS_DIR=${ROOT_DIR}/tools
# Base directory for downloads and builds on /dev/shmem
BASE_DIR="/dev/shm/papi_scorep_build"
mkdir -p "$BASE_DIR"

source ${ROOT_DIR}/source_env.cpp_cpu

# -------------------------------
# Score-P configuration
# -------------------------------
SCOREP_VERSION="9.0"
SCOREP_TARBALL_URL="https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/scorep-${SCOREP_VERSION}/scorep-${SCOREP_VERSION}.tar.gz"
SCOREP_TARBALL_NAME="scorep-${SCOREP_VERSION}.tar.gz"
SCOREP_SRC_DIR="${BASE_DIR}/scorep_clang"
SCOREP_INSTALL_DIR="${TOOLS_DIR}/scorep_clang"

# -----------------------------------------
# Step 0: Download and extract Score-P source
# -----------------------------------------
if [ ! -d "${SCOREP_SRC_DIR}" ]; then
  echo "Downloading and extracting Score-P (${SCOREP_VERSION})..."
  mkdir -p "${SCOREP_SRC_DIR}"
  cd "$BASE_DIR"
  wget "$SCOREP_TARBALL_URL" -O "$SCOREP_TARBALL_NAME"
  tar -xf "$SCOREP_TARBALL_NAME" -C "${SCOREP_SRC_DIR}" --strip-components=1
  cd "${SCRIPT_DIR}"
else
  echo "Score-P source already exists at ${SCOREP_SRC_DIR}"
fi

# -----------------------------------------
# Step 1: Build and Install Score-P with Clang
# -----------------------------------------
if [ ! -d "${SCOREP_INSTALL_DIR}" ]; then
  echo "Building and installing Score-P..."
  mkdir -p "${SCOREP_SRC_DIR}/build"
  cd "${SCOREP_SRC_DIR}/build"
  ../configure --prefix="${SCOREP_INSTALL_DIR}"  \
    --without-shmem \
    --without-mpi \
    --with-libgotcha=download \
    --with-libunwind=download \
    --with-libbfd=download \
    --with-nocross-compiler-suite=clang

  make -j 8
  make install
  cd "${SCRIPT_DIR}"
else
  echo "Score-P already installed at ${SCOREP_INSTALL_DIR}"
fi

echo "Build Successful"

