"""
Shared fixtures and configuration for benchmark suite.
"""

import pytest
import numpy as np
import os
from pathlib import Path


# Benchmark data size configurations
DATA_SIZES = {
    'tiny': 4,
    'small': 16,
    'medium': 64,
    'large': 256,
    'xlarge': 1024,
    'xxlarge': 4096,
}

# Batch size configurations
BATCH_SIZES = [1, 10, 50, 100, 500, 1000, 5000, 10000]

# Qubit count configurations
QUBIT_COUNTS = [4, 8, 16, 32, 64]

# Data dimension configurations
DATA_DIMENSIONS = [8, 16, 32, 64, 128, 256, 512, 1024]


@pytest.fixture
def benchmark_data_dir():
    """Directory for storing benchmark data and results."""
    data_dir = Path(__file__).parent / "reports"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def random_seed():
    """Fixed seed for reproducible benchmarks."""
    return 42


@pytest.fixture(params=['tiny', 'small', 'medium', 'large', 'xlarge'])
def data_size_name(request):
    """Parametrized fixture for data size categories."""
    return request.param


@pytest.fixture(params=BATCH_SIZES)
def batch_size(request):
    """Parametrized fixture for batch sizes."""
    return request.param


@pytest.fixture(params=QUBIT_COUNTS)
def qubit_count(request):
    """Parametrized fixture for qubit counts."""
    return request.param


@pytest.fixture(params=DATA_DIMENSIONS[:4])  # Smaller dimensions for faster testing
def data_dimension(request):
    """Parametrized fixture for data dimensions."""
    return request.param


@pytest.fixture
def tiny_data(random_seed):
    """Tiny data for micro-benchmarks."""
    np.random.seed(random_seed)
    return np.random.random(DATA_SIZES['tiny'])


@pytest.fixture
def small_data(random_seed):
    """Small data for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random(DATA_SIZES['small'])


@pytest.fixture
def medium_data(random_seed):
    """Medium data for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random(DATA_SIZES['medium'])


@pytest.fixture
def large_data(random_seed):
    """Large data for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random(DATA_SIZES['large'])


@pytest.fixture
def xlarge_data(random_seed):
    """Extra large data for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random(DATA_SIZES['xlarge'])


@pytest.fixture
def small_batch(random_seed):
    """Small batch for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random((10, DATA_SIZES['small']))


@pytest.fixture
def medium_batch(random_seed):
    """Medium batch for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random((100, DATA_SIZES['medium']))


@pytest.fixture
def large_batch(random_seed):
    """Large batch for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random((1000, DATA_SIZES['large']))


@pytest.fixture
def xlarge_batch(random_seed):
    """Extra large batch for benchmarks."""
    np.random.seed(random_seed)
    return np.random.random((5000, DATA_SIZES['large']))


@pytest.fixture
def batch_generator(random_seed):
    """Generator function for creating batches of different sizes."""
    def make_batch(batch_size, data_dim, seed=random_seed):
        np.random.seed(seed)
        return np.random.random((batch_size, data_dim))
    return make_batch


@pytest.fixture
def data_generator(random_seed):
    """Generator function for creating data of different sizes."""
    def make_data(size, seed=random_seed):
        np.random.seed(seed)
        return np.random.random(size)
    return make_data


@pytest.fixture
def numpy_baseline():
    """Provide NumPy baseline implementation for comparison."""
    def numpy_encode(data, n_qubits):
        """NumPy baseline for angle encoding."""
        two_pi = 2.0 * np.pi
        result = np.zeros(n_qubits, dtype=np.float64)
        for i in range(min(len(data), n_qubits)):
            result[i] = data[i] * two_pi
        return result

    def numpy_encode_batch(batch_data, n_qubits):
        """NumPy baseline for batch angle encoding."""
        two_pi = 2.0 * np.pi
        batch_size, data_dim = batch_data.shape
        result = np.zeros((batch_size, n_qubits), dtype=np.float64)

        for b in range(batch_size):
            for i in range(min(data_dim, n_qubits)):
                result[b, i] = batch_data[b, i] * two_pi

        return result

    return {
        'encode': numpy_encode,
        'encode_batch': numpy_encode_batch,
    }


@pytest.fixture(scope="session")
def simd_info():
    """Cache SIMD information for all benchmarks."""
    try:
        import simd_angle_encoder as sae
        return sae.simd_info()
    except ImportError:
        return "SIMD info not available"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "micro: Micro-benchmarks")
    config.addinivalue_line("markers", "macro: Macro-benchmarks")
    config.addinivalue_line("markers", "comparative: Comparative benchmarks")
    config.addinivalue_line("markers", "scalability: Scalability benchmarks")
    config.addinivalue_line("markers", "regression: Regression detection benchmarks")


@pytest.fixture(autouse=True)
def skip_benchmarks_if_no_simd(request):
    """Skip benchmarks if SIMD encoder is not available."""
    if request.node.get_closest_marker('benchmark'):
        try:
            import simd_angle_encoder as sae
            # Test that we can actually call the function
            test_data = np.array([0.5])
            _ = sae.encode(test_data, 1)
        except (ImportError, Exception):
            pytest.skip("SIMD encoder not available or not built")
