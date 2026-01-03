#!/bin/bash
# Setup script for pre-commit hooks

set -e

echo "Setting up pre-commit hooks for SIMD Angle Encoder..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Install pre-commit hooks
echo "Installing git hooks..."
pre-commit install

# Install pre-commit commit-msg hook (optional)
pre-commit install --hook-type commit-msg

echo ""
echo "Pre-commit hooks installed successfully!"
echo ""
echo "The following hooks will now run automatically on each commit:"
echo "  - Rust formatting (cargo fmt)"
echo "  - Rust linting (cargo clippy)"
echo "  - Python formatting (black)"
echo "  - Python import sorting (isort)"
echo "  - Python linting (flake8)"
echo "  - YAML/TOML/JSON validation"
echo "  - File checks (trailing whitespace, large files, etc.)"
echo ""
echo "To run all hooks manually:"
echo "  pre-commit run --all-files"
echo ""
echo "To run specific hook:"
echo "  pre-commit run rust-fmt --all-files"
echo ""
echo "To skip hooks for a commit (not recommended):"
echo "  git commit --no-verify -m 'your message'"
