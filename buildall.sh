#!/usr/bin/env bash
# buildall.sh — Build llama.cpp as a shared library, install it, then compile
# the go-do binary.  Run this script once on a new machine; re-run it to
# rebuild after updating llama.cpp.
#
# Usage:
#   ./buildall.sh              # auto-detect everything
#   LLAMA_BACKEND=CUDA ./buildall.sh   # force a specific backend
#
# Supported backend names (for the LLAMA_BACKEND env override):
#   CUDA | HIP | VULKAN | METAL | CPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

echo "========================================"
echo "  go-do build script"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Detect GPU backend
# ---------------------------------------------------------------------------

detect_backend() {
    # macOS → Metal (always, even with discrete GPUs)
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "METAL"; return
    fi

    # NVIDIA CUDA Toolkit (nvcc in PATH or /usr/local/cuda present)
    if command -v nvcc &>/dev/null || [ -d "/usr/local/cuda/bin" ]; then
        echo "CUDA"; return
    fi

    # AMD ROCm / HIP
    if command -v hipcc &>/dev/null || [ -d "/opt/rocm/bin" ]; then
        echo "HIP"; return
    fi

    # Vulkan (headers + loader library)
    local has_headers=0
    local has_loader=0
    if pkg-config --exists vulkan 2>/dev/null \
       || [ -f "/usr/include/vulkan/vulkan.h" ] \
       || [ -f "/usr/local/include/vulkan/vulkan.h" ]; then
        has_headers=1
    fi
    if ldconfig -p 2>/dev/null | grep -q "libvulkan\.so" \
       || find /usr/lib* /usr/local/lib* -maxdepth 2 \
              -name "libvulkan*.so*" 2>/dev/null | grep -q .; then
        has_loader=1
    fi
    if [[ $has_headers -eq 1 && $has_loader -eq 1 ]]; then
        echo "VULKAN"; return
    fi

    echo "CPU"
}

BACKEND="${LLAMA_BACKEND:-$(detect_backend)}"
echo "Detected backend: $BACKEND"

CMAKE_GPU_FLAGS=""
case "$BACKEND" in
    CUDA)   CMAKE_GPU_FLAGS="-DGGML_CUDA=ON" ;;
    HIP)    CMAKE_GPU_FLAGS="-DGGML_HIP=ON" ;;
    VULKAN) CMAKE_GPU_FLAGS="-DGGML_VULKAN=ON" ;;
    METAL)  CMAKE_GPU_FLAGS="-DGGML_METAL=ON" ;;
    CPU)    CMAKE_GPU_FLAGS="" ;;
    *)
        echo "Warning: unknown backend '$BACKEND', falling back to CPU."
        CMAKE_GPU_FLAGS=""
        ;;
esac

# Optionally enable OpenBLAS for CPU-path acceleration
CMAKE_BLAS_FLAGS=""
if pkg-config --exists openblas 2>/dev/null \
   || [ -f "/usr/include/openblas/cblas.h" ] \
   || [ -f "/usr/local/include/openblas/cblas.h" ]; then
    CMAKE_BLAS_FLAGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
    echo "OpenBLAS detected — enabling BLAS acceleration"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Clone llama.cpp if not already present
# ---------------------------------------------------------------------------

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    echo ""
else
    echo "llama.cpp directory found at: $LLAMA_DIR"
    echo "(Run 'git -C llama.cpp pull' to update it before rebuilding.)"
    echo ""
fi

# ---------------------------------------------------------------------------
# 3. Build llama.cpp as a shared library
# ---------------------------------------------------------------------------

echo "Building llama.cpp as a shared library..."
echo "  Backend flags : ${CMAKE_GPU_FLAGS:-<none>}"
echo "  BLAS flags    : ${CMAKE_BLAS_FLAGS:-<none>}"
echo ""

cd "$LLAMA_DIR"

# shellcheck disable=SC2086
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_C_FLAGS=-march=native" \
    "-DCMAKE_CXX_FLAGS=-march=native" \
    $CMAKE_GPU_FLAGS \
    $CMAKE_BLAS_FLAGS
# Note: -march=native produces a binary optimised for the current CPU micro-
# architecture and will not run on machines with older or different CPUs.
# If you need a portable binary, replace -march=native with -march=x86-64-v3
# (or remove it entirely for a generic build).

cmake --build build --config Release -j"$NPROC"
echo ""

# ---------------------------------------------------------------------------
# 4. Install to system paths
# ---------------------------------------------------------------------------

echo "Installing llama.cpp to system paths (requires sudo)..."
sudo cmake --install build
if [[ "$(uname -s)" == "Linux" ]] && command -v ldconfig >/dev/null 2>&1; then
    sudo ldconfig
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Set up shared model directory
# ---------------------------------------------------------------------------

echo "Setting up shared model directory at /usr/share/models ..."
sudo mkdir -p /usr/share/models
# Allow the 'users' group to read models; ignore error if group doesn't exist.
# Permissions are 755 (owner can write, everyone else can read/execute).
# To add a model without sudo, add yourself to the 'users' group first:
#   sudo usermod -aG users "$USER" && newgrp users
sudo chgrp -R users /usr/share/models 2>/dev/null || true
sudo chmod -R 755 /usr/share/models
echo ""

# ---------------------------------------------------------------------------
# 6. Build go-do
# ---------------------------------------------------------------------------

echo "Compiling go-do..."
cd "$SCRIPT_DIR"
CGO_ENABLED=1 go build -o go-do .
echo ""

# ---------------------------------------------------------------------------
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Place a GGUF model at /usr/share/models/default.gguf"
echo "     (or set LLAMA_MODEL_PATH to its path)"
echo ""
echo "  2. Run go-do:"
echo "     ./go-do 'List all files in this directory over 1 GB'"
echo "     ./go-do --dry-run 'Figure out what changed and write a commit message'"
echo "     ./go-do --yes 'Remove all .pyc files recursively'"
echo ""
