# Testing Quick Reference Guide

## Quick Start

### Install Dependencies
```bash
pip install pytest pytest-cov hypothesis numpy
```

### Run All Tests
```bash
cd /Users/syahriza/data/kubitto/simd-angle-encoder
python -m pytest
```

### Run by Category
```bash
# Unit tests only (47 tests)
python -m pytest -m unit -v

# Integration tests only (27 tests)
python -m pytest -m integration -v

# Property-based tests only (20 tests)
python -m pytest -m property -v

# Skip slow tests
python -m pytest -m "not slow"
```

### Run Specific Test File
```bash
# Angle encoding unit tests
python -m pytest tests/unit/test_angle_encoding.py -v

# Batch processing unit tests
python -m pytest tests/unit/test_batch_processing.py -v

# NumPy integration unit tests
python -m pytest tests/unit/test_numpy_integration.py -v

# Python-Rust bridge integration tests
python -m pytest tests/integration/test_python_rust_bridge.py -v

# End-to-end integration tests
python -m pytest tests/integration/test_end_to_end.py -v

# Property-based tests
python -m pytest tests/property/test_encoding_properties.py -v
```

### Run Specific Test
```bash
python -m pytest tests/unit/test_angle_encoding.py::TestAngleEncoding::test_basic_encoding -v
```

### Run with Coverage
```bash
# HTML coverage report (opens in browser)
python -m pytest --cov=simd_angle_encoder --cov-report=html
open htmlcov/index.html

# Terminal coverage report
python -m pytest --cov=simd_angle_encoder --cov-report=term-missing
```

### Run with Verbose Output
```bash
python -m pytest -v --tb=short
```

### Use Convenience Script
```bash
chmod +x run_tests.sh
./run_tests.sh
```

## Test Structure

```
tests/
├── unit/                      # Unit tests (47 tests)
│   ├── test_angle_encoding.py     # Core encoding functionality
│   ├── test_batch_processing.py    # Batch processing
│   └── test_numpy_integration.py   # NumPy integration
├── integration/               # Integration tests (27 tests)
│   ├── test_python_rust_bridge.py  # Python-Rust FFI
│   └── test_end_to_end.py          # End-to-end workflows
├── property/                 # Property-based tests (20 tests)
│   └── test_encoding_properties.py # Invariants and properties
└── conftest.py               # Shared fixtures
```

## Test Categories

### Unit Tests (47 tests)
- Test individual functions in isolation
- Fast execution (< 0.1s each)
- Cover all code paths and edge cases
- Marker: `@pytest.mark.unit`

### Integration Tests (27 tests)
- Test component interactions
- Python-Rust FFI boundary testing
- End-to-end workflow validation
- Marker: `@pytest.mark.integration`

### Property-Based Tests (20 tests)
- Use Hypothesis framework
- Test mathematical invariants
- Generate 50-100 examples per test
- Marker: `@pytest.mark.property`

### Slow Tests (3 tests)
- Large-scale performance tests
- Marked with `@pytest.mark.slow`
- Excluded by default with `-m "not slow"`

## Test Statistics

- **Total Tests:** 94
- **Unit Tests:** 47 (50%)
- **Integration Tests:** 27 (29%)
- **Property-Based Tests:** 20 (21%)
- **Total Lines of Test Code:** ~1,696
- **Reusable Fixtures:** 13

## Key Fixtures

Available in `tests/conftest.py`:

- `random_data` - Random test data (128 elements)
- `small_batch` - Small batch (10x32)
- `data_size` - Parametrized sizes (8, 16, 32, 64, 128)
- `n_qubits` - Qubit count (10)
- `single_value` - Single element
- `zero_array` - Array of zeros
- `ones_array` - Array of ones
- `known_data` - Deterministic data
- `batch_known_data` - Batch deterministic data
- `batch_size` - Parametrized batch sizes (1, 2, 4, 8, 16)
- `qubit_count` - Parametrized qubit counts (5, 10, 15, 20)
- `large_array` - Large array (10,000 elements)
- `large_batch` - Large batch (100x1000)

## Numerical Testing Standards

- **Tolerance:** rtol=1e-10 for high precision
- **Range:** [0, 1] inputs → [0, 2π] outputs
- **Extremes:** 1e-300 to 1e300 value ranges
- **Special Values:** NaN and Inf tested

## Common Issues

### Import Error
```
ImportError: No module named 'simd_angle_encoder'
```
**Solution:** Build the Rust extension first
```bash
cd /Users/syahriza/data/kubitto/simd-angle-encoder
./build.sh
```

### Missing Dependencies
```
ImportError: No module named 'pytest'
```
**Solution:** Install test dependencies
```bash
pip install pytest pytest-cov hypothesis
```

### Coverage Not Showing
**Solution:** Install pytest-cov
```bash
pip install pytest-cov
```

## Test Writing Guidelines

### Unit Test Template
```python
import pytest
import numpy as np
from simd_angle_encoder import encode

@pytest.mark.unit
class TestFeature:
    """Test suite for feature."""

    def test_specific_behavior(self, fixture):
        """Test description."""
        # Arrange
        input_data = fixture

        # Act
        result = encode(input_data, n_qubits=10)

        # Assert
        assert result.shape == (10,)
        assert np.allclose(result, expected, rtol=1e-10)
```

### Property Test Template
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100))
def test_invariant_holds(data):
    """Property description."""
    result = encode(data, n_qubits=50)
    assert invariant_condition(result)
```

## Continuous Integration

To add CI/CD:

1. Create `.github/workflows/test.yml`
2. Add test matrix (Python versions, OS)
3. Run pytest with coverage
4. Upload coverage to Codecov

Example workflow:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install pytest pytest-cov hypothesis
      - run: pytest --cov=simd_angle_encoder
```

## Additional Resources

- Full report: `agents/agent-reports/P1-TASK-001_testing_infrastructure_report.md`
- pytest documentation: https://docs.pytest.org/
- Hypothesis documentation: https://hypothesis.works/
- pytest-cov documentation: https://pytest-cov.readthedocs.io/

## Support

For issues or questions about testing:
1. Check the full report for detailed explanations
2. Review test files for examples
3. Consult pytest and Hypothesis documentation
