"""
Batch encoding benchmarks for SIMD angle encoder.

Tests performance of encode_batch() function across various batch sizes and dimensions.
"""

import pytest
import numpy as np
import simd_angle_encoder as sae


@pytest.mark.benchmark(group="batch-encode-single", min_rounds=100)
@pytest.mark.micro
class TestEncodeBatchSingle:
    """Benchmark batch encode() function with single batch."""

    def test_encode_batch_single_row(self, benchmark):
        """Benchmark encoding a single row (batch_size=1)."""
        data = np.array([[0.1, 0.3, 0.5, 0.7]])

        result = benchmark(sae.encode_batch, data, 4)

        assert result.shape == (1, 4)
        expected = data * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_batch_small_8x4(self, benchmark):
        """Benchmark encoding small batch (8x4)."""
        np.random.seed(42)
        data = np.random.random((8, 4))

        result = benchmark(sae.encode_batch, data, 4)

        assert result.shape == (8, 4)

    def test_encode_batch_medium_32x16(self, benchmark):
        """Benchmark encoding medium batch (32x16)."""
        np.random.seed(42)
        data = np.random.random((32, 16))

        result = benchmark(sae.encode_batch, data, 16)

        assert result.shape == (32, 16)

    def test_encode_batch_large_128x32(self, benchmark):
        """Benchmark encoding large batch (128x32)."""
        np.random.seed(42)
        data = np.random.random((128, 32))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (128, 32)


@pytest.mark.benchmark(group="batch-encode-size", min_rounds=50)
@pytest.mark.macro
class TestEncodeBatchSize:
    """Benchmark encode_batch() with different batch sizes."""

    def test_encode_batch_size_1(self, benchmark):
        """Benchmark batch_size=1."""
        np.random.seed(42)
        data = np.random.random((1, 32))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (1, 32)

    def test_encode_batch_size_10(self, benchmark):
        """Benchmark batch_size=10."""
        np.random.seed(42)
        data = np.random.random((10, 32))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (10, 32)

    def test_encode_batch_size_50(self, benchmark):
        """Benchmark batch_size=50."""
        np.random.seed(42)
        data = np.random.random((50, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (50, 64)

    def test_encode_batch_size_100(self, benchmark):
        """Benchmark batch_size=100."""
        np.random.seed(42)
        data = np.random.random((100, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (100, 64)

    def test_encode_batch_size_500(self, benchmark):
        """Benchmark batch_size=500."""
        np.random.seed(42)
        data = np.random.random((500, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (500, 64)

    def test_encode_batch_size_1000(self, benchmark):
        """Benchmark batch_size=1000."""
        np.random.seed(42)
        data = np.random.random((1000, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (1000, 64)

    def test_encode_batch_size_5000(self, benchmark):
        """Benchmark batch_size=5000."""
        np.random.seed(42)
        data = np.random.random((5000, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (5000, 64)

    def test_encode_batch_size_10000(self, benchmark):
        """Benchmark batch_size=10000."""
        np.random.seed(42)
        data = np.random.random((10000, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (10000, 64)


@pytest.mark.benchmark(group="batch-encode-dimensions", min_rounds=50)
@pytest.mark.scalability
class TestEncodeBatchDimensions:
    """Benchmark encode_batch() with different data dimensions."""

    def test_encode_batch_dim_8(self, benchmark):
        """Benchmark with data_dim=8."""
        np.random.seed(42)
        data = np.random.random((100, 8))

        result = benchmark(sae.encode_batch, data, 8)

        assert result.shape == (100, 8)

    def test_encode_batch_dim_16(self, benchmark):
        """Benchmark with data_dim=16."""
        np.random.seed(42)
        data = np.random.random((100, 16))

        result = benchmark(sae.encode_batch, data, 16)

        assert result.shape == (100, 16)

    def test_encode_batch_dim_32(self, benchmark):
        """Benchmark with data_dim=32."""
        np.random.seed(42)
        data = np.random.random((100, 32))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (100, 32)

    def test_encode_batch_dim_64(self, benchmark):
        """Benchmark with data_dim=64."""
        np.random.seed(42)
        data = np.random.random((100, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (100, 64)

    def test_encode_batch_dim_128(self, benchmark):
        """Benchmark with data_dim=128."""
        np.random.seed(42)
        data = np.random.random((100, 128))

        result = benchmark(sae.encode_batch, data, 128)

        assert result.shape == (100, 128)

    def test_encode_batch_dim_256(self, benchmark):
        """Benchmark with data_dim=256."""
        np.random.seed(42)
        data = np.random.random((100, 256))

        result = benchmark(sae.encode_batch, data, 256)

        assert result.shape == (100, 256)

    def test_encode_batch_dim_512(self, benchmark):
        """Benchmark with data_dim=512."""
        np.random.seed(42)
        data = np.random.random((100, 512))

        result = benchmark(sae.encode_batch, data, 512)

        assert result.shape == (100, 512)


@pytest.mark.benchmark(group="batch-encode-qubits", min_rounds=50)
@pytest.mark.micro
class TestEncodeBatchQubits:
    """Benchmark encode_batch() with different qubit counts."""

    def test_encode_batch_qubits_4(self, benchmark):
        """Benchmark with n_qubits=4."""
        np.random.seed(42)
        data = np.random.random((100, 32))

        result = benchmark(sae.encode_batch, data, 4)

        assert result.shape == (100, 4)

    def test_encode_batch_qubits_8(self, benchmark):
        """Benchmark with n_qubits=8."""
        np.random.seed(42)
        data = np.random.random((100, 32))

        result = benchmark(sae.encode_batch, data, 8)

        assert result.shape == (100, 8)

    def test_encode_batch_qubits_16(self, benchmark):
        """Benchmark with n_qubits=16."""
        np.random.seed(42)
        data = np.random.random((100, 32))

        result = benchmark(sae.encode_batch, data, 16)

        assert result.shape == (100, 16)

    def test_encode_batch_qubits_32(self, benchmark):
        """Benchmark with n_qubits=32."""
        np.random.seed(42)
        data = np.random.random((100, 64))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (100, 32)

    def test_encode_batch_qubits_64(self, benchmark):
        """Benchmark with n_qubits=64."""
        np.random.seed(42)
        data = np.random.random((100, 128))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (100, 64)


@pytest.mark.benchmark(group="batch-encode-mixed", min_rounds=50)
@pytest.mark.macro
class TestEncodeBatchMixed:
    """Benchmark encode_batch() with mixed batch and dimension sizes."""

    def test_encode_batch_small_batch_large_dim(self, benchmark):
        """Benchmark small batch with large dimensions."""
        np.random.seed(42)
        data = np.random.random((10, 256))

        result = benchmark(sae.encode_batch, data, 256)

        assert result.shape == (10, 256)

    def test_encode_batch_large_batch_small_dim(self, benchmark):
        """Benchmark large batch with small dimensions."""
        np.random.seed(42)
        data = np.random.random((1000, 8))

        result = benchmark(sae.encode_batch, data, 8)

        assert result.shape == (1000, 8)

    def test_encode_batch_balanced(self, benchmark):
        """Benchmark balanced batch and dimensions."""
        np.random.seed(42)
        data = np.random.random((100, 100))

        result = benchmark(sae.encode_batch, data, 100)

        assert result.shape == (100, 100)


@pytest.mark.benchmark(group="batch-encode-edge", min_rounds=50)
@pytest.mark.micro
class TestEncodeBatchEdgeCases:
    """Benchmark encode_batch() edge cases."""

    def test_encode_batch_dim_smaller_than_qubits(self, benchmark):
        """Benchmark when data_dim < n_qubits."""
        np.random.seed(42)
        data = np.random.random((50, 8))

        result = benchmark(sae.encode_batch, data, 16)

        assert result.shape == (50, 16)
        # Last 8 columns should be zeros
        assert np.all(result[:, 8:] == 0)

    def test_encode_batch_dim_larger_than_qubits(self, benchmark):
        """Benchmark when data_dim > n_qubits."""
        np.random.seed(42)
        data = np.random.random((50, 128))

        result = benchmark(sae.encode_batch, data, 16)

        assert result.shape == (50, 16)

    def test_encode_batch_zeros(self, benchmark):
        """Benchmark encoding all zeros."""
        data = np.zeros((50, 32))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (50, 32)
        assert np.all(result == 0)

    def test_encode_batch_ones(self, benchmark):
        """Benchmark encoding all ones."""
        data = np.ones((50, 32))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (50, 32)
        expected = np.full((50, 32), 2 * np.pi)
        assert np.allclose(result, expected, rtol=1e-10)


@pytest.mark.benchmark(group="batch-encode-scaling", min_rounds=30)
@pytest.mark.scalability
class TestEncodeBatchScaling:
    """Test scalability of encode_batch() with increasing sizes."""

    @pytest.mark.parametrize("batch_size,dim", [
        (10, 16),
        (50, 32),
        (100, 64),
        (500, 128),
        (1000, 256),
        (5000, 256),
    ])
    def test_encode_batch_scaling(self, benchmark, batch_size, dim):
        """Benchmark encode_batch() scaling with batch size and dimension."""
        np.random.seed(42)
        data = np.random.random((batch_size, dim))

        result = benchmark(sae.encode_batch, data, dim)

        assert result.shape == (batch_size, dim)


@pytest.mark.benchmark(group="batch-encode-throughput", min_rounds=20)
@pytest.mark.macro
class TestEncodeBatchThroughput:
    """Benchmark encode_batch() throughput (elements per second)."""

    def test_encode_batch_throughput_small(self, benchmark):
        """Benchmark throughput for small batches."""
        np.random.seed(42)
        data = np.random.random((100, 32))

        result = benchmark(sae.encode_batch, data, 32)

        assert result.shape == (100, 32)
        # Throughput will be calculated by pytest-benchmark

    def test_encode_batch_throughput_medium(self, benchmark):
        """Benchmark throughput for medium batches."""
        np.random.seed(42)
        data = np.random.random((500, 64))

        result = benchmark(sae.encode_batch, data, 64)

        assert result.shape == (500, 64)

    def test_encode_batch_throughput_large(self, benchmark):
        """Benchmark throughput for large batches."""
        np.random.seed(42)
        data = np.random.random((5000, 128))

        result = benchmark(sae.encode_batch, data, 128)

        assert result.shape == (5000, 128)
