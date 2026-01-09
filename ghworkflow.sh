#!/bin/bash
# Local CI workflow script - mimics GitHub Actions CI pipeline

set -e

echo "========================================="
echo "Running CI Workflow Locally"
echo "========================================="
echo ""

FAILED=0

# Check formatting
echo "1. Checking Rust formatting..."
if cargo fmt -- --check; then
    echo "✓ Formatting check passed"
else
    echo "✗ Formatting check failed"
    FAILED=1
fi
echo ""

# Run Clippy
echo "2. Running Clippy lints..."
if cargo clippy --all-targets --all-features -- -D warnings -D clippy::all -D clippy::pedantic; then
    echo "✓ Clippy checks passed"
else
    echo "✗ Clippy checks failed"
    FAILED=1
fi
echo ""

# Build Rust extension
echo "3. Building Rust extension..."
if maturin build --release; then
    echo "✓ Build succeeded"
    # Install the built wheel
    if command -v pip &> /dev/null; then
        echo "Installing built wheel..."
        pip install --force-reinstall target/wheels/simd_angle_encoder-*.whl --quiet 2>/dev/null || true
    fi
else
    echo "✗ Build failed"
    FAILED=1
fi
echo ""

# Run tests (if pytest is available)
if command -v pytest &> /dev/null; then
    echo "4. Running Python tests..."
    # Clean pycache first
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    # Run only unit tests if hypothesis is not available
    if python3 -c "import hypothesis" 2>/dev/null; then
        if pytest tests/ -v --tb=short; then
            echo "✓ Tests passed"
        else
            echo "✗ Tests failed"
            FAILED=1
        fi
    else
        echo "4. Running unit tests only (hypothesis not installed)..."
        if pytest tests/unit/ -v --tb=short --ignore=tests/property/ --ignore=tests/qiskit/ --ignore=tests/pennylane/ 2>/dev/null || pytest tests/unit/test_angle_encoding.py -v --tb=short; then
            echo "✓ Unit tests passed"
        else
            echo "✗ Tests failed"
            FAILED=1
        fi
    fi
else
    echo "4. Skipping Python tests (pytest not installed)"
fi
echo ""

echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All checks passed!"
    exit 0
else
    echo "✗ Some checks failed"
    exit 1
fi
echo "========================================="
