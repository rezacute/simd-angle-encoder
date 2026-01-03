"""
Single encoding benchmarks for SIMD angle encoder.

Tests performance of encode() function across various data sizes and qubit counts.
"""

import pytest
import numpy as np
import simd_angle_encoder as sae


@pytest.mark.benchmark(group="encode-single", min_rounds=100)
@pytest.mark.micro
class TestEncodeSingle:
    """Benchmark single encode() function performance."""

    def test_encode_single_value(self, benchmark):
        """Benchmark encoding a single value."""
        data = np.array([0.5])

        result = benchmark(sae.encode, data, 1)

        assert len(result) == 1
        assert result[0] == pytest.approx(0.5 * 2 * np.pi, rel=1e-10)

    def test_encode_tiny_data_4(self, benchmark):
        """Benchmark encoding tiny data (4 elements)."""
        data = np.array([0.1, 0.3, 0.5, 0.7])

        result = benchmark(sae.encode, data, 4)

        assert len(result) == 4
        expected = data * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_small_data_16(self, benchmark):
        """Benchmark encoding small data (16 elements)."""
        np.random.seed(42)
        data = np.random.random(16)

        result = benchmark(sae.encode, data, 16)

        assert len(result) == 16
        expected = data * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_medium_data_64(self, benchmark):
        """Benchmark encoding medium data (64 elements)."""
        np.random.seed(42)
        data = np.random.random(64)

        result = benchmark(sae.encode, data, 64)

        assert len(result) == 64
        expected = data[:64] * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_large_data_256(self, benchmark):
        """Benchmark encoding large data (256 elements)."""
        np.random.seed(42)
        data = np.random.random(256)

        result = benchmark(sae.encode, data, 256)

        assert len(result) == 256
        expected = data[:256] * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_xlarge_data_1024(self, benchmark):
        """Benchmark encoding extra large data (1024 elements)."""
        np.random.seed(42)
        data = np.random.random(1024)

        result = benchmark(sae.encode, data, 1024)

        assert len(result) == 1024
        expected = data[:1024] * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_xxlarge_data_4096(self, benchmark):
        """Benchmark encoding extra extra large data (4096 elements)."""
        np.random.seed(42)
        data = np.random.random(4096)

        result = benchmark(sae.encode, data, 4096)

        assert len(result) == 4096
        expected = data[:4096] * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)


@pytest.mark.benchmark(group="encode-qubits", min_rounds=100)
@pytest.mark.micro
class TestEncodeQubits:
    """Benchmark encode() with different qubit counts."""

    def test_encode_4_qubits(self, benchmark):
        """Benchmark encoding with 4 qubits."""
        np.random.seed(42)
        data = np.random.random(32)

        result = benchmark(sae.encode, data, 4)

        assert len(result) == 4

    def test_encode_8_qubits(self, benchmark):
        """Benchmark encoding with 8 qubits."""
        np.random.seed(42)
        data = np.random.random(32)

        result = benchmark(sae.encode, data, 8)

        assert len(result) == 8

    def test_encode_16_qubits(self, benchmark):
        """Benchmark encoding with 16 qubits."""
        np.random.seed(42)
        data = np.random.random(64)

        result = benchmark(sae.encode, data, 16)

        assert len(result) == 16

    def test_encode_32_qubits(self, benchmark):
        """Benchmark encoding with 32 qubits."""
        np.random.seed(42)
        data = np.random.random(128)

        result = benchmark(sae.encode, data, 32)

        assert len(result) == 32

    def test_encode_64_qubits(self, benchmark):
        """Benchmark encoding with 64 qubits."""
        np.random.seed(42)
        data = np.random.random(256)

        result = benchmark(sae.encode, data, 64)

        assert len(result) == 64


@pytest.mark.benchmark(group="encode-edge", min_rounds=50)
@pytest.mark.micro
class TestEncodeEdgeCases:
    """Benchmark encode() edge cases."""

    def test_encode_data_smaller_than_qubits(self, benchmark):
        """Benchmark when data is smaller than n_qubits."""
        np.random.seed(42)
        data = np.random.random(8)

        result = benchmark(sae.encode, data, 16)

        assert len(result) == 16
        # Last 8 elements should be zeros
        assert np.all(result[8:] == 0)

    def test_encode_data_larger_than_qubits(self, benchmark):
        """Benchmark when data is larger than n_qubits."""
        np.random.seed(42)
        data = np.random.random(128)

        result = benchmark(sae.encode, data, 16)

        assert len(result) == 16
        # Should only use first 16 elements
        expected = data[:16] * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_zeros(self, benchmark):
        """Benchmark encoding all zeros."""
        data = np.zeros(64)

        result = benchmark(sae.encode, data, 64)

        assert len(result) == 64
        assert np.all(result == 0)

    def test_encode_ones(self, benchmark):
        """Benchmark encoding all ones."""
        data = np.ones(64)

        result = benchmark(sae.encode, data, 64)

        assert len(result) == 64
        expected = np.full(64, 2 * np.pi)
        assert np.allclose(result, expected, rtol=1e-10)

    def test_encode_mixed_values(self, benchmark):
        """Benchmark encoding mixed values including edge values."""
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.5])

        result = benchmark(sae.encode, data, 8)

        assert len(result) == 8
        expected = data * 2 * np.pi
        assert np.allclose(result, expected, rtol=1e-10)


@pytest.mark.benchmark(group="encode-repeated", min_rounds=200)
@pytest.mark.micro
class TestEncodeRepeated:
    """Benchmark encode() when called repeatedly (simulating real usage)."""

    def test_encode_repeated_small(self, benchmark):
        """Benchmark repeated small encodings."""
        np.random.seed(42)

        def encode_multiple():
            for _ in range(10):
                data = np.random.random(16)
                sae.encode(data, 16)
            return True

        result = benchmark(encode_multiple)

        assert result is True

    def test_encode_repeated_medium(self, benchmark):
        """Benchmark repeated medium encodings."""
        np.random.seed(42)

        def encode_multiple():
            for _ in range(10):
                data = np.random.random(64)
                sae.encode(data, 64)
            return True

        result = benchmark(encode_multiple)

        assert result is True

    def test_encode_repeated_large(self, benchmark):
        """Benchmark repeated large encodings."""
        np.random.seed(42)

        def encode_multiple():
            for _ in range(10):
                data = np.random.random(256)
                sae.encode(data, 256)
            return True

        result = benchmark(encode_multiple)

        assert result is True


@pytest.mark.benchmark(group="encode-scaling", min_rounds=50)
@pytest.mark.scalability
class TestEncodeScaling:
    """Test scalability of encode() with increasing data sizes."""

    @pytest.mark.parametrize("size", [8, 16, 32, 64, 128, 256, 512, 1024])
    def test_encode_scaling(self, benchmark, size):
        """Benchmark encode() scaling with data size."""
        np.random.seed(42)
        data = np.random.random(size)

        result = benchmark(sae.encode, data, size)

        assert len(result) == size
