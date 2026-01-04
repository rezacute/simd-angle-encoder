"""Test NEON implementation for correctness and performance."""

import numpy as np
import pytest
from simd_angle_encoder import encode


class TestNEONValidation:
    """Validate NEON implementation correctness."""

    def test_neon_matches_scalar_for_various_sizes(self):
        """Test that NEON implementation matches scalar for various input sizes."""
        # Test different sizes to hit different code paths
        test_sizes = [1, 2, 3, 4, 5, 8, 16, 32, 33, 64, 100, 128, 256, 512, 1024]

        for size in test_sizes:
            # Create test data
            data = np.random.rand(size).astype(np.float64)
            n_qubits = size

            # Encode using NEON path (when size > 32)
            result = encode(data, n_qubits)

            # Expected result: data * 2π
            expected = data * 2 * np.pi

            # Verify results match
            assert np.allclose(result, expected, rtol=1e-10, atol=1e-10), \
                f"NEON result mismatch for size {size}"

            # Verify output shape
            assert result.shape == (n_qubits,), \
                f"Output shape mismatch for size {size}"

    def test_neon_zero_padding(self):
        """Test that NEON implementation correctly pads with zeros."""
        # Test case: data has fewer elements than n_qubits
        data = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        n_qubits = 10

        result = encode(data, n_qubits)

        # First 3 elements should be data * 2π
        expected_prefix = data * 2 * np.pi
        assert np.allclose(result[:3], expected_prefix, rtol=1e-10)

        # Remaining elements should be zeros
        assert np.all(result[3:] == 0.0), "Expected zero padding"

    def test_neon_edge_cases(self):
        """Test edge cases with special values."""
        # Test with zeros
        data = np.zeros(10, dtype=np.float64)
        result = encode(data, 10)
        assert np.all(result == 0.0), "Zeros should remain zeros"

        # Test with ones
        data = np.ones(10, dtype=np.float64)
        result = encode(data, 10)
        expected = np.full(10, 2 * np.pi)
        assert np.allclose(result, expected, rtol=1e-10), "Ones should become 2π"

        # Test with 0.5 (common case)
        data = np.full(10, 0.5, dtype=np.float64)
        result = encode(data, 10)
        expected = np.full(10, np.pi)
        assert np.allclose(result, expected, rtol=1e-10), "0.5 should become π"

    def test_neon_batch_processing(self):
        """Test that batch encoding works correctly with NEON."""
        # Create batch data
        batch_data = np.random.rand(10, 64).astype(np.float64)
        n_qubits = 64

        # Encode batch
        from simd_angle_encoder import encode_batch
        result = encode_batch(batch_data, n_qubits)

        # Verify shape
        assert result.shape == batch_data.shape, "Batch output shape mismatch"

        # Verify each row
        for i in range(batch_data.shape[0]):
            expected = batch_data[i] * 2 * np.pi
            assert np.allclose(result[i], expected, rtol=1e-10), \
                f"Row {i} mismatch in batch encoding"

    def test_neon_three_tier_strategy(self):
        """Test that three-tier optimization strategy works correctly."""
        # Tier 1: Stack allocation (n_qubits <= 32)
        small_data = np.random.rand(16).astype(np.float64)
        result_small = encode(small_data, 16)
        expected_small = small_data * 2 * np.pi
        assert np.allclose(result_small, expected_small, rtol=1e-10), \
            "Tier 1 (stack allocation) failed"

        # Tier 2: NEON SIMD (n_qubits > 32, data.len() >= 2)
        medium_data = np.random.rand(64).astype(np.float64)
        result_medium = encode(medium_data, 64)
        expected_medium = medium_data * 2 * np.pi
        assert np.allclose(result_medium, expected_medium, rtol=1e-10), \
            "Tier 2 (NEON SIMD) failed"

        # Tier 3: Scalar fallback (data.len() < 2, but this won't happen in practice)
        # Just verify large arrays work
        large_data = np.random.rand(512).astype(np.float64)
        result_large = encode(large_data, 512)
        expected_large = large_data * 2 * np.pi
        assert np.allclose(result_large, expected_large, rtol=1e-10), \
            "Tier 3 (scalar/large array) failed"

    def test_neon_numerical_precision(self):
        """Test numerical precision of NEON implementation."""
        # Use values that exercise floating-point precision
        data = np.array([0.1, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

        result = encode(data, 5)
        expected = data * 2 * np.pi

        # Check ULP (Units in Last Place) - should be very small
        # Allow small floating-point differences (within 4 ULP)
        max_diff = np.max(np.abs(result - expected))
        assert max_diff < 1e-15, f"Numerical precision error: {max_diff}"

        # Verify relative error is small
        max_rel_diff = np.max(np.abs((result - expected) / expected))
        assert max_rel_diff < 1e-15, f"Relative error too large: {max_rel_diff}"
