#!/bin/bash
set -euxo pipefail

# Determine the directory where the script resides
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Base directory for downloads and builds on /dev/shmem
BASE_DIR="/dev/shmem/papi_scorep_build"
mkdir -p "$BASE_DIR"

# -------------------------------
# PAPI configuration
# -------------------------------
PAPI_VERSION="7.2.0b2"
PAPI_TARBALL_URL="https://icl.utk.edu/projects/papi/downloads/papi-${PAPI_VERSION}.tar.gz"
PAPI_TARBALL_NAME="papi-${PAPI_VERSION}.tar.gz"
PAPI_SRC_DIR="${BASE_DIR}/papi_gcc"
PAPI_INSTALL_DIR="${SCRIPT_DIR}/install/papi_gcc"

# -------------------------------
# Score-P configuration
# -------------------------------
SCOREP_VERSION="9.0"
SCOREP_TARBALL_URL="https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/scorep-${SCOREP_VERSION}/scorep-${SCOREP_VERSION}.tar.gz"
SCOREP_TARBALL_NAME="scorep-${SCOREP_VERSION}.tar.gz"
SCOREP_SRC_DIR="${BASE_DIR}/scorep_gcc"
SCOREP_INSTALL_DIR="${SCRIPT_DIR}/install/scorep_gcc"

# Create the install directory (inside the script directory) if it does not exist
mkdir -p "${SCRIPT_DIR}/install"

# -----------------------------------------
# Step 0: Download and extract PAPI source
# -----------------------------------------
if [ ! -d "${PAPI_SRC_DIR}" ]; then
  echo "Downloading and extracting PAPI (${PAPI_VERSION})..."
  mkdir -p "${PAPI_SRC_DIR}"
  cd "$BASE_DIR"
  wget "$PAPI_TARBALL_URL" -O "$PAPI_TARBALL_NAME"
  tar -xf "$PAPI_TARBALL_NAME" -C "${PAPI_SRC_DIR}" --strip-components=1
  cd "${SCRIPT_DIR}"
else
  echo "PAPI source already exists at ${PAPI_SRC_DIR}"
fi

# -----------------------------------------
# Step 1: Build and Install PAPI with GCC
# -----------------------------------------
if [ ! -d "${PAPI_INSTALL_DIR}" ]; then
  echo "Building and installing PAPI..."
  cd "${PAPI_SRC_DIR}/src"
  ./configure --prefix="${PAPI_INSTALL_DIR}"
  make -j 4
  make install
  cd "${SCRIPT_DIR}"
else
  echo "PAPI already installed at ${PAPI_INSTALL_DIR}"
fi

# -----------------------------------------
# Step 2: Download and extract Score-P source
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
# Step 3: Build and Install Score-P with GCC
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
    --with-papi="${PAPI_INSTALL_DIR}" \
    --with-nocross-compiler-suite=gcc

  make -j 4
  make install
  cd "${SCRIPT_DIR}"
else
  echo "Score-P already installed at ${SCOREP_INSTALL_DIR}"
fi

echo "Build Successful"

