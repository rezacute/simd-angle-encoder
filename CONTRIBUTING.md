# Contributing to SIMD Angle Encoder

Thank you for your interest in contributing to the SIMD Angle Encoder project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Testing Guidelines](#testing-guidelines)
- [CI/CD Pipeline](#cicd-pipeline)
- [Pull Request Process](#pull-request-process)
- [Code Review Process](#code-review-process)

## Development Setup

### Prerequisites

- Python 3.8 or later
- Rust toolchain (stable)
- Git
- Make (optional, for convenience scripts)

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hybriq/simd_angle_encoder.git
   cd simd_angle_encoder
   ```

2. **Install Python dependencies**:
   ```bash
   pip install maturin[patchelf] pytest pytest-cov numpy hypothesis black isort flake8 mypy bandit
   ```

3. **Install Rust toolchain** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

4. **Build the project**:
   ```bash
   maturin develop --release
   ```

5. **Set up pre-commit hooks** (recommended):
   ```bash
   ./setup-hooks.sh
   # or manually:
   pip install pre-commit
   pre-commit install
   ```

6. **Run tests to verify setup**:
   ```bash
   pytest tests/ -v
   ```

## Code Quality Standards

### Rust Code

All Rust code must adhere to the following standards:

1. **Formatting**:
   ```bash
   cargo fmt
   ```

2. **Linting** (pass all Clippy checks):
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```

3. **Testing** (all tests must pass):
   ```bash
   cargo test
   ```

4. **Documentation**:
   - All public functions must have doc comments
   - Include examples in doc comments where appropriate
   - Run `cargo doc` to verify documentation builds

### Python Code

All Python code must adhere to the following standards:

1. **Formatting** (Black):
   ```bash
   black python/ tests/ benchmarks/
   ```

2. **Import Sorting** (isort):
   ```bash
   isort python/ tests/ benchmarks/
   ```

3. **Linting** (flake8):
   ```bash
   flake8 python/ tests/ benchmarks/
   ```

4. **Type Checking** (mypy):
   ```bash
   mypy python/
   ```

5. **Security** (bandit):
   ```bash
   bandit -r python/
   ```

## Testing Guidelines

### Test Structure

The project uses a three-tier testing approach:

1. **Unit Tests** (`tests/unit/`): Test individual functions and components
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **Property-Based Tests** (`tests/property/`): Test invariants with random inputs

### Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Property-based tests only
pytest tests/property/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=simd_angle_encoder --cov-report=html
```

Run specific test file:
```bash
pytest tests/unit/test_angle_encoding.py -v
```

Run specific test:
```bash
pytest tests/unit/test_angle_encoding.py::TestAngleEncoding::test_basic_encoding -v
```

### Writing Tests

When writing new tests, follow these guidelines:

1. **Unit Tests**:
   - Test individual functions in isolation
   - Use fixtures from `conftest.py` when appropriate
   - Test edge cases (empty arrays, single elements, large inputs)
   - Mark with `@pytest.mark.unit`

2. **Integration Tests**:
   - Test interactions between components
   - Use real dependencies
   - Mark with `@pytest.mark.integration`

3. **Property-Based Tests**:
   - Identify invariants that must always hold
   - Use Hypothesis for generating test cases
   - Mark with `@pytest.mark.property`

Example test structure:
```python
import pytest
import numpy as np
from simd_angle_encoder import encode

@pytest.mark.unit
class TestFeature:
    """Descriptive test class."""

    def test_specific_behavior(self):
        """Test description explaining what is tested."""
        # Arrange - Set up test data
        input_data = np.array([0.5, 0.75, 1.0])

        # Act - Execute function under test
        result = encode(input_data, n_qubits=10)

        # Assert - Verify expected outcome
        assert result.shape == (10,)
        assert np.allclose(result, expected, rtol=1e-10)
```

### Coverage Goals

- Target line coverage: 90%+
- Target branch coverage: 85%+
- Target function coverage: 95%+

View coverage report:
```bash
pytest tests/ --cov=simd_angle_encoder --cov-report=html
open htmlcov/index.html
```

## CI/CD Pipeline

### Overview

The project uses GitHub Actions for continuous integration and deployment. The pipeline consists of:

1. **Rust Quality Checks**:
   - Code formatting verification (`cargo fmt --check`)
   - Linting with Clippy
   - Unit and integration tests
   - Release build verification

2. **Python Tests**:
   - Multi-version testing (Python 3.8-3.12)
   - Multi-platform testing (Ubuntu, macOS, Windows)
   - Coverage reporting
   - Upload to Codecov

3. **Integration Tests**:
   - End-to-end functionality tests
   - Python-Rust FFI boundary tests

4. **Property-Based Tests**:
   - Invariant testing with Hypothesis
   - Statistical property verification

5. **Security Scanning**:
   - Rust dependency audit
   - Outdated dependency checks

### CI Matrix

The CI pipeline tests across:

| OS | Python Versions | Status |
|----|----------------|--------|
| Ubuntu Latest | 3.8, 3.9, 3.10, 3.11, 3.12 | ✅ |
| macOS Latest | 3.9, 3.10, 3.11, 3.12 | ✅ |
| Windows Latest | 3.9, 3.10, 3.11, 3.12 | ✅ |

### Viewing CI Results

- **Workflow Status**: Check the badge in the README
- **Detailed Logs**: Go to Actions tab in GitHub
- **Coverage Reports**: View on Codecov

### Local CI Simulation

To run the same checks as CI locally:

1. **Rust checks**:
   ```bash
   cargo fmt -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --verbose
   ```

2. **Python checks**:
   ```bash
   pytest tests/ --cov=simd_angle_encoder --cov-report=xml --cov-report=term-missing
   ```

3. **Pre-commit hooks** (same as CI):
   ```bash
   pre-commit run --all-files
   ```

## Pull Request Process

### Before Submitting a PR

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**:
   - Write clear, concise commit messages
   - Follow commit message conventions:
     - `feat: add new batch encoding feature`
     - `fix: resolve memory leak in encode function`
     - `test: add property tests for encoding invariants`
     - `docs: update installation instructions`

3. **Run all quality checks**:
   ```bash
   # Format and lint Rust
   cargo fmt
   cargo clippy --all-targets --all-features -- -D warnings

   # Format and lint Python
   black python/ tests/ benchmarks/
   isort python/ tests/ benchmarks/
   flake8 python/ tests/ benchmarks/

   # Run tests
   pytest tests/ -v

   # Run pre-commit hooks
   pre-commit run --all-files
   ```

4. **Ensure tests pass** and maintain coverage standards

5. **Update documentation** if needed

### Submitting a PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Link to related issues (e.g., "Fixes #123")
   - Screenshots for UI changes (if applicable)
   - List of breaking changes (if any)

3. **Fill in the PR template** (if provided)

4. **Wait for CI checks** to complete
   - All checks must pass before merge
   - Fix any failures that arise

### PR Review Process

1. **Automated Checks**:
   - All CI checks must pass
   - Code coverage must not decrease significantly
   - No new security vulnerabilities

2. **Code Review**:
   - At least one maintainer approval required
   - Address all review comments
   - Make requested changes or provide justification

3. **Testing**:
   - Reviewer may request additional tests
   - Ensure edge cases are covered
   - Verify backward compatibility

4. **Merge**:
   - Maintainer merges after approval
   - Squash and merge preferred for clean history
   - Delete branch after merge

## Code Review Process

### For Contributors

- Respond to review comments promptly
- Either make requested changes or discuss alternative approaches
- Mark conversations as resolved when addressed
- Keep an open mind to feedback

### For Reviewers

- Be constructive and respectful
- Explain the reasoning behind suggestions
- Approve when satisfied with changes
- Test critical functionality if needed

### Review Criteria

When reviewing code, check for:

1. **Correctness**: Does the code work as intended?
2. **Testing**: Are there adequate tests?
3. **Documentation**: Is the code well-documented?
4. **Style**: Does it follow project conventions?
5. **Performance**: Are there obvious performance issues?
6. **Security**: Are there potential security concerns?
7. **Maintainability**: Is the code easy to understand and maintain?

## Getting Help

If you need help:

1. **Check existing issues**: Someone may have asked the same question
2. **Create an issue**: Describe your question or problem clearly
3. **Join discussions**: Engage in conversations about the project
4. **Read documentation**: Check inline documentation and examples

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in the CONTRIBUTORS section of the documentation. Thank you for your contributions!
