"""
Scalability benchmarks for SIMD angle encoder.

Analyzes performance scaling characteristics across different data sizes,
batch sizes, and dimensions to ensure linear or better scaling.
"""

import pytest
import numpy as np
import simd_angle_encoder as sae


@pytest.mark.benchmark(group="scalability-data-size", min_rounds=50)
@pytest.mark.scalability
class TestDataSizeScaling:
    """Test encode() scaling with increasing data size."""

    @pytest.mark.parametrize("size", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    def test_encode_data_size_scaling(self, benchmark, size):
        """Benchmark encode() across different data sizes."""
        np.random.seed(42)
        data = np.random.random(size)

        result = benchmark(sae.encode, data, size)

        assert len(result) == size

    @pytest.mark.parametrize("n_qubits", [4, 8, 16, 32, 64, 128, 256])
    def test_encode_qubit_scaling(self, benchmark, n_qubits):
        """Benchmark encode() across different qubit counts."""
        np.random.seed(42)
        # Keep data size constant, vary qubits
        data = np.random.random(256)

        result = benchmark(sae.encode, data, n_qubits)

        assert len(result) == n_qubits


@pytest.mark.benchmark(group="scalability-batch-size", min_rounds=30)
@pytest.mark.scalability
class TestBatchSizeScaling:
    """Test encode_batch() scaling with increasing batch size."""

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    def test_batch_size_scaling(self, benchmark, batch_size):
        """Benchmark encode_batch() across different batch sizes."""
        np.random.seed(42)
        # Keep dimension constant, vary batch size
        batch_data = np.random.random((batch_size, 64))

        result = benchmark(sae.encode_batch, batch_data, 64)

        assert result.shape == (batch_size, 64)

    @pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256, 512, 1024])
    def test_batch_dimension_scaling(self, benchmark, dim):
        """Benchmark encode_batch() across different data dimensions."""
        np.random.seed(42)
        # Keep batch size constant, vary dimension
        batch_data = np.random.random((100, dim))

        result = benchmark(sae.encode_batch, batch_data, dim)

        assert result.shape == (100, dim)


@pytest.mark.benchmark(group="scalability-mixed", min_rounds=30)
@pytest.mark.scalability
class TestMixedScaling:
    """Test scaling with varying both batch size and dimensions."""

    @pytest.mark.parametrize("batch_size,dim", [
        (10, 16),
        (50, 32),
        (100, 64),
        (500, 128),
        (1000, 256),
        (5000, 256),
        (10000, 128),
        (5000, 512),
        (1000, 1024),
    ])
    def test_batch_mixed_scaling(self, benchmark, batch_size, dim):
        """Benchmark encode_batch() with various batch/dimension combinations."""
        np.random.seed(42)
        batch_data = np.random.random((batch_size, dim))

        result = benchmark(sae.encode_batch, batch_data, dim)

        assert result.shape == (batch_size, dim)


@pytest.mark.benchmark(group="scalability-throughput", min_rounds=20)
@pytest.mark.scalability
class TestThroughputScaling:
    """Test throughput scaling (elements processed per second)."""

    @pytest.mark.parametrize("batch_size,dim", [
        (100, 64),     # 6,400 elements
        (500, 64),     # 32,000 elements
        (1000, 128),   # 128,000 elements
        (5000, 128),   # 640,000 elements
        (10000, 256),  # 2,560,000 elements
    ])
    def test_throughput_large_batches(self, benchmark, batch_size, dim):
        """Benchmark throughput for very large batches."""
        np.random.seed(42)
        batch_data = np.random.random((batch_size, dim))

        result = benchmark(sae.encode_batch, batch_data, dim)

        assert result.shape == (batch_size, dim)
        # Throughput (elements/sec) will be shown in benchmark output


@pytest.mark.benchmark(group="scalability-memory", min_rounds=20)
@pytest.mark.scalability
class TestMemoryEfficiency:
    """Test memory efficiency and cache behavior."""

    def test_encode_small_memory_footprint(self, benchmark):
        """Benchmark with minimal memory usage."""
        np.random.seed(42)
        data = np.random.random(16)

        result = benchmark(sae.encode, data, 16)

        assert len(result) == 16

    def test_encode_medium_memory_footprint(self, benchmark):
        """Benchmark with moderate memory usage."""
        np.random.seed(42)
        data = np.random.random(256)

        result = benchmark(sae.encode, data, 256)

        assert len(result) == 256

    def test_encode_large_memory_footprint(self, benchmark):
        """Benchmark with high memory usage."""
        np.random.seed(42)
        data = np.random.random(4096)

        result = benchmark(sae.encode, data, 4096)

        assert len(result) == 4096

    def test_encode_very_large_memory_footprint(self, benchmark):
        """Benchmark with very high memory usage."""
        np.random.seed(42)
        data = np.random.random(16384)

        result = benchmark(sae.encode, data, 16384)

        assert len(result) == 16384

    def test_batch_cache_friendly(self, benchmark):
        """Benchmark cache-friendly access pattern (small, repeated)."""
        np.random.seed(42)

        def repeated_small_batches():
            for _ in range(10):
                batch_data = np.random.random((100, 32))
                sae.encode_batch(batch_data, 32)
            return True

        result = benchmark(repeated_small_batches)

        assert result is True

    def test_batch_cache_unfriendly(self, benchmark):
        """Benchmark cache-unfriendly access pattern (large, single)."""
        np.random.seed(42)
        # Large batch that may exceed cache
        batch_data = np.random.random((10000, 256))

        result = benchmark(sae.encode_batch, batch_data, 256)

        assert result.shape == (10000, 256)


@pytest.mark.benchmark(group="scalability-linearity", min_rounds=30)
@pytest.mark.scalability
class TestLinearScaling:
    """Verify linear scaling characteristics."""

    @pytest.mark.parametrize("multiplier", [1, 2, 4, 8, 16])
    def test_encode_linear_scaling(self, benchmark, multiplier):
        """Test that encode() scales linearly with data size."""
        base_size = 64
        size = base_size * multiplier
        np.random.seed(42)
        data = np.random.random(size)

        result = benchmark(sae.encode, data, size)

        assert len(result) == size
        # Time should scale roughly linearly with multiplier

    @pytest.mark.parametrize("multiplier", [1, 2, 4, 8, 16])
    def test_batch_linear_scaling(self, benchmark, multiplier):
        """Test that encode_batch() scales linearly with batch size."""
        base_batch = 100
        batch_size = base_batch * multiplier
        np.random.seed(42)
        batch_data = np.random.random((batch_size, 64))

        result = benchmark(sae.encode_batch, batch_data, 64)

        assert result.shape == (batch_size, 64)
        # Time should scale roughly linearly with multiplier


@pytest.mark.scalability
class TestScalingAnalysis:
    """Analyze and report scaling characteristics."""

    def test_analyze_single_scaling(self):
        """Analyze scaling of encode() across data sizes."""
        import time

        sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
        times = []

        for size in sizes:
            np.random.seed(42)
            data = np.random.random(size)

            # Warmup
            for _ in range(10):
                _ = sae.encode(data, size)

            # Measure
            start = time.time()
            n_runs = max(10, 10000 // size)  # More runs for smaller sizes
            for _ in range(n_runs):
                _ = sae.encode(data, size)
            elapsed = time.time() - start

            avg_time_ms = (elapsed / n_runs) * 1000
            times.append(avg_time_ms)

            print(f"\nEncode size={size:4d}: {avg_time_ms:.6f} ms")

        # Check approximate linearity
        # Time should increase proportionally with size
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]

            # Allow for some overhead, but should be roughly linear
            # Time ratio should be between 0.5*size_ratio and 2.0*size_ratio
            assert time_ratio > 0.5 * size_ratio, \
                f"Scaling too slow: size_ratio={size_ratio:.2f}, time_ratio={time_ratio:.2f}"
            assert time_ratio < 2.0 * size_ratio, \
                f"Scaling too fast: size_ratio={size_ratio:.2f}, time_ratio={time_ratio:.2f}"

    def test_analyze_batch_scaling(self):
        """Analyze scaling of encode_batch() across batch sizes."""
        import time

        batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        dim = 64
        times = []

        for batch_size in batch_sizes:
            np.random.seed(42)
            batch_data = np.random.random((batch_size, dim))

            # Warmup
            for _ in range(5):
                _ = sae.encode_batch(batch_data, dim)

            # Measure
            start = time.time()
            n_runs = max(5, 1000 // batch_size)  # More runs for smaller batches
            for _ in range(n_runs):
                _ = sae.encode_batch(batch_data, dim)
            elapsed = time.time() - start

            avg_time_ms = (elapsed / n_runs) * 1000
            times.append(avg_time_ms)

            print(f"\nBatch size={batch_size:5d}, dim={dim}: {avg_time_ms:.6f} ms")

        # Check approximate linearity
        # Time should increase proportionally with batch_size
        for i in range(1, len(batch_sizes)):
            batch_ratio = batch_sizes[i] / batch_sizes[i-1]
            time_ratio = times[i] / times[i-1]

            # Allow for some overhead, but should be roughly linear
            # Time ratio should be between 0.5*batch_ratio and 2.0*batch_ratio
            assert time_ratio > 0.5 * batch_ratio, \
                f"Scaling too slow: batch_ratio={batch_ratio:.2f}, time_ratio={time_ratio:.2f}"
            assert time_ratio < 2.0 * batch_ratio, \
                f"Scaling too fast: batch_ratio={batch_ratio:.2f}, time_ratio={time_ratio:.2f}"

    def test_analyze_dimension_scaling(self):
        """Analyze scaling of encode_batch() across dimensions."""
        import time

        dims = [8, 16, 32, 64, 128, 256, 512]
        batch_size = 100
        times = []

        for dim in dims:
            np.random.seed(42)
            batch_data = np.random.random((batch_size, dim))

            # Warmup
            for _ in range(10):
                _ = sae.encode_batch(batch_data, dim)

            # Measure
            start = time.time()
            n_runs = max(10, 5000 // dim)  # More runs for smaller dims
            for _ in range(n_runs):
                _ = sae.encode_batch(batch_data, dim)
            elapsed = time.time() - start

            avg_time_ms = (elapsed / n_runs) * 1000
            times.append(avg_time_ms)

            print(f"\nDim={dim:3d}, batch_size={batch_size}: {avg_time_ms:.6f} ms")

        # Check approximate linearity
        # Time should increase proportionally with dim
        for i in range(1, len(dims)):
            dim_ratio = dims[i] / dims[i-1]
            time_ratio = times[i] / times[i-1]

            # Allow for some overhead, but should be roughly linear
            # Time ratio should be between 0.5*dim_ratio and 2.0*dim_ratio
            assert time_ratio > 0.5 * dim_ratio, \
                f"Scaling too slow: dim_ratio={dim_ratio:.2f}, time_ratio={time_ratio:.2f}"
            assert time_ratio < 2.0 * dim_ratio, \
                f"Scaling too fast: dim_ratio={dim_ratio:.2f}, time_ratio={time_ratio:.2f}"


@pytest.mark.benchmark(group="scalability-real-world", min_rounds=20)
@pytest.mark.macro
class TestRealWorldWorkloads:
    """Benchmark realistic quantum ML workloads."""

    def test_vqc_small_training_batch(self, benchmark):
        """Simulate small VQC training batch."""
        np.random.seed(42)
        # Typical VQC small training batch
        batch_data = np.random.random((50, 32))

        result = benchmark(sae.encode_batch, batch_data, 32)

        assert result.shape == (50, 32)

    def test_vqc_medium_training_batch(self, benchmark):
        """Simulate medium VQC training batch."""
        np.random.seed(42)
        # Typical VQC medium training batch
        batch_data = np.random.random((128, 64))

        result = benchmark(sae.encode_batch, batch_data, 64)

        assert result.shape == (128, 64)

    def test_vqc_large_training_batch(self, benchmark):
        """Simulate large VQC training batch."""
        np.random.seed(42)
        # Typical VQC large training batch
        batch_data = np.random.random((500, 128))

        result = benchmark(sae.encode_batch, batch_data, 128)

        assert result.shape == (500, 128)

    def test_quantum_data_preprocessing(self, benchmark):
        """Simulate quantum data preprocessing pipeline."""
        np.random.seed(42)

        def preprocess_pipeline():
            # Simulate multiple preprocessing steps
            batch_data = np.random.random((100, 64))

            # Step 1: Normalize (simulated)
            normalized = batch_data / np.max(batch_data)

            # Step 2: Angle encode (actual operation we're benchmarking)
            encoded = sae.encode_batch(normalized, 64)

            return encoded

        result = benchmark(preprocess_pipeline)

        assert result.shape == (100, 64)

    def test_inference_workload(self, benchmark):
        """Simulate inference workload (single samples)."""
        np.random.seed(42)

        def inference_samples():
            results = []
            for _ in range(50):
                # Single sample for inference
                sample = np.random.random(64)
                encoded = sae.encode(sample, 64)
                results.append(encoded)
            return results

        result = benchmark(inference_samples)

        assert len(result) == 50
