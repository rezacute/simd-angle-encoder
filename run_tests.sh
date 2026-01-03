#!/bin/bash
# Script to run pytest tests

echo "========================================="
echo "Running SIMD Angle Encoder Test Suite"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pytest.ini" ]; then
    echo "Error: pytest.ini not found. Please run from project root."
    exit 1
fi

# Install pytest and hypothesis if needed
echo "Installing test dependencies..."
pip install pytest pytest-cov hypothesis -q

echo ""
echo "Running all tests..."
echo ""

# Run all tests
python -m pytest tests/ -v --tb=short $@

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
