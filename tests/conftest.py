"""
Pytest configuration and shared fixtures for SIMD Angle Encoder tests.
"""

import pytest
import numpy as np


@pytest.fixture
def random_data():
    """Fixture providing random test data."""
    np.random.seed(42)  # For reproducibility
    return np.random.random(128)


@pytest.fixture
def small_batch():
    """Fixture providing small batch for testing."""
    np.random.seed(42)
    return np.random.random((10, 32))


@pytest.fixture(params=[8, 16, 32, 64, 128])
def data_size(request):
    """Parametrized fixture for different data sizes."""
    return request.param


@pytest.fixture
def n_qubits():
    """Fixture for qubit count."""
    return 10


@pytest.fixture
def single_value():
    """Fixture providing a single value for testing."""
    return np.array([0.5])


@pytest.fixture
def zero_array():
    """Fixture providing an array of zeros."""
    return np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def ones_array():
    """Fixture providing an array of ones."""
    return np.array([1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def known_data():
    """Fixture providing known data for deterministic testing."""
    return np.array([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.fixture
def batch_known_data():
    """Fixture providing batch of known data for deterministic testing."""
    return np.array([
        [0.0, 0.25, 0.5, 0.75],
        [1.0, 0.0, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5]
    ])


@pytest.fixture(params=[1, 2, 4, 8, 16])
def batch_size(request):
    """Parametrized fixture for different batch sizes."""
    return request.param


@pytest.fixture(params=[5, 10, 15, 20])
def qubit_count(request):
    """Parametrized fixture for different qubit counts."""
    return request.param


@pytest.fixture
def large_array():
    """Fixture providing large array for performance testing."""
    np.random.seed(42)
    return np.random.random(10000)


@pytest.fixture
def large_batch():
    """Fixture providing large batch for performance testing."""
    np.random.seed(42)
    return np.random.random((100, 1000))
