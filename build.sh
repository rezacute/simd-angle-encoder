#!/bin/bash

set -e  # Exit on error

echo "===> Building SIMD Angle Encoder..."

# Check for Python 3.13
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "===> Python 3.13 detected, setting PyO3 compatibility flag"
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
fi

# Check for virtual environment conflicts
if [[ -n "$VIRTUAL_ENV" && -n "$CONDA_PREFIX" ]]; then
    echo "===> Both VIRTUAL_ENV and CONDA_PREFIX are set, temporarily unsetting for build"
    VIRTUAL_ENV_BACKUP=$VIRTUAL_ENV
    CONDA_PREFIX_BACKUP=$CONDA_PREFIX
    unset VIRTUAL_ENV
    unset CONDA_PREFIX
    RESTORE_ENV=1
else
    RESTORE_ENV=0
fi

# Build the package
echo "===> Running maturin build..."
if ! maturin build --release; then
    echo "===> maturin build failed, trying with development dependencies..."
    pip install maturin
    maturin build --release
fi

# Restore environment variables if needed
if [[ $RESTORE_ENV -eq 1 ]]; then
    export VIRTUAL_ENV=$VIRTUAL_ENV_BACKUP
    export CONDA_PREFIX=$CONDA_PREFIX_BACKUP
fi

# Install the built wheel
echo "===> Installing package..."
pip install --force-reinstall $(ls -t target/wheels/*.whl | head -1)

echo "===> Build complete!"
echo "You can now import the module with: from simd_angle_encoder import encode, encode_batch" 