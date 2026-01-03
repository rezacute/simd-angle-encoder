# Development Quick Start Guide

This guide will help you quickly set up a development environment and start contributing to the SIMD Angle Encoder project.

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/hybriq/simd_angle_encoder.git
cd simd_angle_encoder

# Install Python and Rust dependencies
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks

```bash
./setup-hooks.sh
```

### 3. Build the Project

```bash
maturin develop --release
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=simd_angle_encoder --cov-report=html
```

That's it! You're ready to develop.

## Common Development Tasks

### Running Quality Checks

```bash
# Rust checks
cargo fmt --check          # Check formatting
cargo clippy               # Run linter
cargo test                 # Run tests

# Python checks
black . --check            # Check formatting
isort . --check            # Check import sorting
flake8 .                   # Run linter
mypy python/               # Type checking
bandit -r python/          # Security checks

# All checks at once (via pre-commit)
pre-commit run --all-files
```

### Formatting Code

```bash
# Rust
cargo fmt

# Python
black .
isort .
```

### Running Specific Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Property-based tests only
pytest tests/property/ -v

# Specific test file
pytest tests/unit/test_angle_encoding.py -v

# Specific test
pytest tests/unit/test_angle_encoding.py::TestAngleEncoding::test_basic_encoding -v
```

### Running Benchmarks

```bash
# Python benchmarks
pytest benchmarks/python/test_bench_encode.py --benchmark-only -v
pytest benchmarks/python/test_bench_encode_batch.py --benchmark-only -v

# Rust benchmarks
cargo bench
```

### Building for Release

```bash
# Python wheel
maturin build --release

# Rust binary
cargo build --release
```

## CI/CD Local Simulation

To simulate the CI/CD pipeline locally:

```bash
# 1. Rust quality checks
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --verbose

# 2. Python tests with coverage
pytest tests/ --cov=simd_angle_encoder --cov-report=xml --cov-report=term-missing -v

# 3. Run pre-commit hooks
pre-commit run --all-files
```

## Debugging Tips

### Viewing SIMD Information

```python
from simd_angle_encoder import simd_info
print(simd_info())
```

### Debugging Rust Code

```bash
# Build with debug symbols
maturin develop

# Run with Rust backtrace
RUST_BACKTRACE=1 pytest tests/ -v
```

### Profiling Python Code

```bash
# Use pytest-benchmark for profiling
pytest tests/ --benchmark-only --benchmark-autosave
```

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`):
   - Test individual functions
   - Fast execution (< 0.1s each)
   - Use fixtures from `conftest.py`

2. **Integration Tests** (`tests/integration/`):
   - Test component interactions
   - Test Python-Rust FFI boundaries

3. **Property-Based Tests** (`tests/property/`):
   - Test invariants with random inputs
   - Use Hypothesis for generation

### Coverage Goals

- Line coverage: 90%+
- Branch coverage: 85%+
- Function coverage: 95%+

## Workflow Example

Here's a typical workflow for making a change:

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make your changes
# ... edit code ...

# 3. Format and lint
cargo fmt
black .
isort .

# 4. Run tests
pytest tests/ -v

# 5. Run pre-commit hooks
pre-commit run --all-files

# 6. Commit and push
git add .
git commit -m "feat: add my new feature"
git push origin feature/my-feature

# 7. Create pull request on GitHub
```

## Getting Help

- Check [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines
- Review existing issues on GitHub
- Check test files for examples
- Read inline documentation in source code

## IDE Recommendations

### VS Code

Install these extensions:
- `rust-lang.rust-analyzer` - Rust language server
- `ms-python.python` - Python support
- `ms-python.vscode-pylance` - Python IntelliSense
- `tamasfe.even-better-toml` - TOML support

### PyCharm

- Rust plugin is available
- Built-in Python support
- Good pytest integration

## Useful Commands Reference

```bash
# Development
maturin develop              # Build in development mode
maturin develop --release    # Build in release mode

# Testing
pytest tests/ -v             # Run all tests
pytest --cov                # With coverage
pytest -x                   # Stop on first failure
pytest -k "test_basic"      # Run tests matching pattern

# Benchmarks
pytest --benchmark-only      # Run benchmarks only
cargo bench                  # Run Rust benchmarks

# Quality
cargo fmt                    # Format Rust code
cargo clippy                 # Lint Rust code
black .                      # Format Python code
flake8 .                     # Lint Python code

# Documentation
cargo doc --open            # Build and open Rust docs
```

Happy coding!
