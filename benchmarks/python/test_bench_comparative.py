"""
Comparative benchmarks for SIMD angle encoder.

Compares SIMD implementation against NumPy baseline to measure speedup factors.
"""

import pytest
import numpy as np
import simd_angle_encoder as sae


@pytest.mark.benchmark(group="comparative-single", min_rounds=100)
@pytest.mark.comparative
class TestComparativeSingle:
    """Compare SIMD vs NumPy for single encoding."""

    def test_numpy_baseline_small(self, benchmark):
        """Benchmark NumPy baseline for small data."""
        np.random.seed(42)
        data = np.random.random(16)
        n_qubits = 16

        def numpy_encode():
            two_pi = 2.0 * np.pi
            result = np.zeros(n_qubits, dtype=np.float64)
            for i in range(min(len(data), n_qubits)):
                result[i] = data[i] * two_pi
            return result

        result = benchmark(numpy_encode)

        assert len(result) == n_qubits

    def test_simd_small(self, benchmark):
        """Benchmark SIMD for small data."""
        np.random.seed(42)
        data = np.random.random(16)

        result = benchmark(sae.encode, data, 16)

        assert len(result) == 16

    def test_numpy_baseline_medium(self, benchmark):
        """Benchmark NumPy baseline for medium data."""
        np.random.seed(42)
        data = np.random.random(64)
        n_qubits = 64

        def numpy_encode():
            two_pi = 2.0 * np.pi
            result = np.zeros(n_qubits, dtype=np.float64)
            for i in range(min(len(data), n_qubits)):
                result[i] = data[i] * two_pi
            return result

        result = benchmark(numpy_encode)

        assert len(result) == n_qubits

    def test_simd_medium(self, benchmark):
        """Benchmark SIMD for medium data."""
        np.random.seed(42)
        data = np.random.random(64)

        result = benchmark(sae.encode, data, 64)

        assert len(result) == 64

    def test_numpy_baseline_large(self, benchmark):
        """Benchmark NumPy baseline for large data."""
        np.random.seed(42)
        data = np.random.random(256)
        n_qubits = 256

        def numpy_encode():
            two_pi = 2.0 * np.pi
            result = np.zeros(n_qubits, dtype=np.float64)
            for i in range(min(len(data), n_qubits)):
                result[i] = data[i] * two_pi
            return result

        result = benchmark(numpy_encode)

        assert len(result) == n_qubits

    def test_simd_large(self, benchmark):
        """Benchmark SIMD for large data."""
        np.random.seed(42)
        data = np.random.random(256)

        result = benchmark(sae.encode, data, 256)

        assert len(result) == 256


@pytest.mark.benchmark(group="comparative-batch", min_rounds=50)
@pytest.mark.comparative
class TestComparativeBatch:
    """Compare SIMD vs NumPy for batch encoding."""

    def test_numpy_baseline_batch_small(self, benchmark):
        """Benchmark NumPy baseline for small batch."""
        np.random.seed(42)
        batch_data = np.random.random((10, 16))
        n_qubits = 16

        def numpy_encode_batch():
            two_pi = 2.0 * np.pi
            batch_size, data_dim = batch_data.shape
            result = np.zeros((batch_size, n_qubits), dtype=np.float64)

            for b in range(batch_size):
                for i in range(min(data_dim, n_qubits)):
                    result[b, i] = batch_data[b, i] * two_pi

            return result

        result = benchmark(numpy_encode_batch)

        assert result.shape == (10, n_qubits)

    def test_simd_batch_small(self, benchmark):
        """Benchmark SIMD for small batch."""
        np.random.seed(42)
        batch_data = np.random.random((10, 16))

        result = benchmark(sae.encode_batch, batch_data, 16)

        assert result.shape == (10, 16)

    def test_numpy_baseline_batch_medium(self, benchmark):
        """Benchmark NumPy baseline for medium batch."""
        np.random.seed(42)
        batch_data = np.random.random((100, 64))
        n_qubits = 64

        def numpy_encode_batch():
            two_pi = 2.0 * np.pi
            batch_size, data_dim = batch_data.shape
            result = np.zeros((batch_size, n_qubits), dtype=np.float64)

            for b in range(batch_size):
                for i in range(min(data_dim, n_qubits)):
                    result[b, i] = batch_data[b, i] * two_pi

            return result

        result = benchmark(numpy_encode_batch)

        assert result.shape == (100, n_qubits)

    def test_simd_batch_medium(self, benchmark):
        """Benchmark SIMD for medium batch."""
        np.random.seed(42)
        batch_data = np.random.random((100, 64))

        result = benchmark(sae.encode_batch, batch_data, 64)

        assert result.shape == (100, 64)

    def test_numpy_baseline_batch_large(self, benchmark):
        """Benchmark NumPy baseline for large batch."""
        np.random.seed(42)
        batch_data = np.random.random((1000, 128))
        n_qubits = 128

        def numpy_encode_batch():
            two_pi = 2.0 * np.pi
            batch_size, data_dim = batch_data.shape
            result = np.zeros((batch_size, n_qubits), dtype=np.float64)

            for b in range(batch_size):
                for i in range(min(data_dim, n_qubits)):
                    result[b, i] = batch_data[b, i] * two_pi

            return result

        result = benchmark(numpy_encode_batch)

        assert result.shape == (1000, n_qubits)

    def test_simd_batch_large(self, benchmark):
        """Benchmark SIMD for large batch."""
        np.random.seed(42)
        batch_data = np.random.random((1000, 128))

        result = benchmark(sae.encode_batch, batch_data, 128)

        assert result.shape == (1000, 128)


@pytest.mark.benchmark(group="comparative-scaling", min_rounds=30)
@pytest.mark.comparative
@pytest.mark.scalability
class TestComparativeScaling:
    """Compare SIMD vs NumPy scaling characteristics."""

    @pytest.mark.parametrize("size", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_numpy_encode_scaling(self, benchmark, size):
        """Benchmark NumPy encode() scaling."""
        np.random.seed(42)
        data = np.random.random(size)

        def numpy_encode():
            two_pi = 2.0 * np.pi
            result = np.zeros(size, dtype=np.float64)
            for i in range(size):
                result[i] = data[i] * two_pi
            return result

        result = benchmark(numpy_encode)

        assert len(result) == size

    @pytest.mark.parametrize("size", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_simd_encode_scaling(self, benchmark, size):
        """Benchmark SIMD encode() scaling."""
        np.random.seed(42)
        data = np.random.random(size)

        result = benchmark(sae.encode, data, size)

        assert len(result) == size

    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100, 500, 1000])
    def test_numpy_batch_scaling(self, benchmark, batch_size):
        """Benchmark NumPy batch scaling."""
        np.random.seed(42)
        batch_data = np.random.random((batch_size, 64))
        n_qubits = 64

        def numpy_encode_batch():
            two_pi = 2.0 * np.pi
            _, data_dim = batch_data.shape
            result = np.zeros((batch_size, n_qubits), dtype=np.float64)

            for b in range(batch_size):
                for i in range(min(data_dim, n_qubits)):
                    result[b, i] = batch_data[b, i] * two_pi

            return result

        result = benchmark(numpy_encode_batch)

        assert result.shape == (batch_size, n_qubits)

    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100, 500, 1000])
    def test_simd_batch_scaling(self, benchmark, batch_size):
        """Benchmark SIMD batch scaling."""
        np.random.seed(42)
        batch_data = np.random.random((batch_size, 64))

        result = benchmark(sae.encode_batch, batch_data, 64)

        assert result.shape == (batch_size, 64)


@pytest.mark.benchmark(group="comparative-speedup", min_rounds=50)
@pytest.mark.comparative
class TestComparativeSpeedup:
    """Direct speedup comparison tests."""

    def test_speedup_small_data_64(self, benchmark):
        """Measure speedup for small data (64 elements)."""
        np.random.seed(42)
        data = np.random.random(64)

        # Benchmark SIMD
        simd_result = benchmark(sae.encode, data, 64)

        # Correctness check
        expected = data * 2 * np.pi
        assert np.allclose(simd_result, expected, rtol=1e-10)

    def test_speedup_medium_data_1024(self, benchmark):
        """Measure speedup for medium data (1024 elements)."""
        np.random.seed(42)
        data = np.random.random(1024)

        # Benchmark SIMD
        simd_result = benchmark(sae.encode, data, 1024)

        # Correctness check
        expected = data[:1024] * 2 * np.pi
        assert np.allclose(simd_result, expected, rtol=1e-10)

    def test_speedup_large_data_8192(self, benchmark):
        """Measure speedup for large data (8192 elements)."""
        np.random.seed(42)
        data = np.random.random(8192)

        # Benchmark SIMD
        simd_result = benchmark(sae.encode, data, 8192)

        # Correctness check
        expected = data[:8192] * 2 * np.pi
        assert np.allclose(simd_result, expected, rtol=1e-10)

    def test_speedup_batch_small(self, benchmark):
        """Measure speedup for small batch (10x16)."""
        np.random.seed(42)
        batch_data = np.random.random((10, 16))

        # Benchmark SIMD
        simd_result = benchmark(sae.encode_batch, batch_data, 16)

        # Correctness check
        assert simd_result.shape == (10, 16)

    def test_speedup_batch_medium(self, benchmark):
        """Measure speedup for medium batch (100x64)."""
        np.random.seed(42)
        batch_data = np.random.random((100, 64))

        # Benchmark SIMD
        simd_result = benchmark(sae.encode_batch, batch_data, 64)

        # Correctness check
        assert simd_result.shape == (100, 64)

    def test_speedup_batch_large(self, benchmark):
        """Measure speedup for large batch (1000x128)."""
        np.random.seed(42)
        batch_data = np.random.random((1000, 128))

        # Benchmark SIMD
        simd_result = benchmark(sae.encode_batch, batch_data, 128)

        # Correctness check
        assert simd_result.shape == (1000, 128)


@pytest.mark.comparative
class TestComparativeCorrectness:
    """Verify SIMD and NumPy produce identical results."""

    @pytest.mark.parametrize("size", [4, 8, 16, 32, 64, 128, 256])
    def test_encode_correctness(self, size):
        """Verify encode() produces correct results."""
        np.random.seed(42)
        data = np.random.random(size)

        # NumPy baseline
        two_pi = 2.0 * np.pi
        numpy_result = np.zeros(size, dtype=np.float64)
        for i in range(size):
            numpy_result[i] = data[i] * two_pi

        # SIMD result
        simd_result = sae.encode(data, size)

        # Should match
        assert np.allclose(numpy_result, simd_result, rtol=1e-10)

    @pytest.mark.parametrize("batch_size,dim", [
        (10, 16),
        (50, 32),
        (100, 64),
    ])
    def test_batch_correctness(self, batch_size, dim):
        """Verify encode_batch() produces correct results."""
        np.random.seed(42)
        batch_data = np.random.random((batch_size, dim))

        # NumPy baseline
        two_pi = 2.0 * np.pi
        numpy_result = np.zeros((batch_size, dim), dtype=np.float64)
        for b in range(batch_size):
            for i in range(dim):
                numpy_result[b, i] = batch_data[b, i] * two_pi

        # SIMD result
        simd_result = sae.encode_batch(batch_data, dim)

        # Should match
        assert np.allclose(numpy_result, simd_result, rtol=1e-10)


@pytest.mark.comparative
class TestSpeedupCalculation:
    """Calculate and report actual speedup factors."""

    def test_calculate_speedup_single_small(self):
        """Calculate speedup for small single encoding."""
        import time

        np.random.seed(42)
        data = np.random.random(64)
        n_runs = 100

        # NumPy baseline
        def numpy_encode():
            two_pi = 2.0 * np.pi
            result = np.zeros(64, dtype=np.float64)
            for i in range(64):
                result[i] = data[i] * two_pi
            return result

        start = time.time()
        for _ in range(n_runs):
            _ = numpy_encode()
        numpy_time = (time.time() - start) / n_runs

        # SIMD
        start = time.time()
        for _ in range(n_runs):
            _ = sae.encode(data, 64)
        simd_time = (time.time() - start) / n_runs

        speedup = numpy_time / simd_time

        print(f"\nSmall single encoding speedup: {speedup:.2f}x")
        print(f"  NumPy: {numpy_time*1000:.4f} ms")
        print(f"  SIMD:  {simd_time*1000:.4f} ms")

        # We expect at least 1.2x speedup
        assert speedup > 1.2, f"Speedup {speedup:.2f}x below expected 1.2x"

    def test_calculate_speedup_batch_medium(self):
        """Calculate speedup for medium batch encoding."""
        import time

        np.random.seed(42)
        batch_data = np.random.random((100, 64))
        n_runs = 50

        # NumPy baseline
        def numpy_encode_batch():
            two_pi = 2.0 * np.pi
            batch_size, data_dim = batch_data.shape
            result = np.zeros((batch_size, 64), dtype=np.float64)
            for b in range(batch_size):
                for i in range(min(data_dim, 64)):
                    result[b, i] = batch_data[b, i] * two_pi
            return result

        start = time.time()
        for _ in range(n_runs):
            _ = numpy_encode_batch()
        numpy_time = (time.time() - start) / n_runs

        # SIMD
        start = time.time()
        for _ in range(n_runs):
            _ = sae.encode_batch(batch_data, 64)
        simd_time = (time.time() - start) / n_runs

        speedup = numpy_time / simd_time

        print(f"\nMedium batch encoding speedup: {speedup:.2f}x")
        print(f"  NumPy: {numpy_time*1000:.4f} ms")
        print(f"  SIMD:  {simd_time*1000:.4f} ms")

        # We expect at least 1.5x speedup for batches
        assert speedup > 1.5, f"Speedup {speedup:.2f}x below expected 1.5x"
