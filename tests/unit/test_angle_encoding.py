"""
Unit tests for angle encoding functionality.

Tests the core encoding functions with various inputs and edge cases.
"""

import pytest
import numpy as np
from simd_angle_encoder import encode


@pytest.mark.unit
class TestAngleEncoding:
    """Test suite for angle encoding functionality."""

    def test_basic_encoding(self, random_data, n_qubits):
        """Test basic encoding of random data."""
        result = encode(random_data, n_qubits)

        # Check output shape
        assert result.shape == (n_qubits,), f"Expected shape ({n_qubits},), got {result.shape}"

        # Check output dtype
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"

        # Check output range [0, 2π]
        assert np.all(result >= 0.0), "All encoded values should be >= 0"
        assert np.all(result <= 2.0 * np.pi), "All encoded values should be <= 2π"

    def test_encoding_range(self, n_qubits):
        """Test that encoding maps [0,1] to [0,2π]."""
        test_data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = encode(test_data, n_qubits)

        # Check first 5 elements (actual encoded values)
        two_pi = 2.0 * np.pi
        expected = test_data * two_pi

        # Only check up to min(len(test_data), n_qubits)
        check_count = min(len(test_data), n_qubits)
        assert np.allclose(result[:check_count], expected[:check_count], rtol=1e-10), \
            f"Encoding mismatch: expected {expected[:check_count]}, got {result[:check_count]}"

    def test_zero_input(self, zero_array, n_qubits):
        """Test encoding of zero values."""
        result = encode(zero_array, n_qubits)

        # Check that first elements are zero (0.0 * 2π = 0.0)
        check_count = min(len(zero_array), n_qubits)
        assert np.allclose(result[:check_count], 0.0, atol=1e-10), \
            f"Zero input should produce zero output, got {result[:check_count]}"

    def test_single_element(self, single_value, n_qubits):
        """Test encoding of a single element."""
        result = encode(single_value, n_qubits)

        # Check shape
        assert result.shape == (n_qubits,), f"Expected shape ({n_qubits},), got {result.shape}"

        # Check first element (0.5 * 2π = π)
        expected_first = 0.5 * 2.0 * np.pi
        assert np.allclose(result[0], expected_first, rtol=1e-10), \
            f"Expected first element {expected_first}, got {result[0]}"

        # Check that remaining elements are padded with zeros
        if n_qubits > 1:
            assert np.allclose(result[1:], 0.0, atol=1e-10), \
                f"Expected padding with zeros, got {result[1:]}"

    def test_size_mismatch(self):
        """Test behavior when data size doesn't match n_qubits."""
        # Data larger than n_qubits
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        n_qubits = 5
        result = encode(data, n_qubits)

        # Should truncate to n_qubits
        assert result.shape == (n_qubits,), f"Expected shape ({n_qubits},), got {result.shape}"

        # Check that first n_qubits are encoded correctly
        two_pi = 2.0 * np.pi
        expected = data[:n_qubits] * two_pi
        assert np.allclose(result, expected, rtol=1e-10), \
            f"Encoding mismatch: expected {expected}, got {result}"

    def test_ones_input(self, ones_array, n_qubits):
        """Test encoding of ones (should map to 2π)."""
        result = encode(ones_array, n_qubits)

        two_pi = 2.0 * np.pi
        check_count = min(len(ones_array), n_qubits)

        # All ones should map to 2π
        assert np.allclose(result[:check_count], two_pi, rtol=1e-10), \
            f"Ones should map to 2π, expected {two_pi}, got {result[:check_count]}"

    def test_known_values(self, known_data):
        """Test encoding with known deterministic values."""
        n_qubits = 10
        result = encode(known_data, n_qubits)

        two_pi = 2.0 * np.pi
        expected = known_data * two_pi

        # Check first 5 elements
        check_count = min(len(known_data), n_qubits)
        assert np.allclose(result[:check_count], expected[:check_count], rtol=1e-10), \
            f"Known values encoding mismatch: expected {expected[:check_count]}, got {result[:check_count]}"

    def test_list_input(self, n_qubits):
        """Test that list inputs are properly converted."""
        data_list = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = encode(data_list, n_qubits)

        # Check it works with lists
        assert result.shape == (n_qubits,), f"Expected shape ({n_qubits},), got {result.shape}"
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"

    def test_2d_input_flattened(self, n_qubits):
        """Test that 2D inputs are flattened."""
        data_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = encode(data_2d, n_qubits)

        # Should flatten and encode
        assert result.shape == (n_qubits,), f"Expected shape ({n_qubits},), got {result.shape}"

    def test_different_dtypes(self, n_qubits):
        """Test that different input dtypes are handled correctly."""
        data_f32 = np.array([0.5, 0.5], dtype=np.float32)
        data_i32 = np.array([1, 1], dtype=np.int32)

        result_f32 = encode(data_f32, n_qubits)
        result_i32 = encode(data_i32, n_qubits)

        # Both should work and produce float64 output
        assert result_f32.dtype == np.float64
        assert result_i32.dtype == np.float64

        # Results should be correct for each input
        two_pi = 2.0 * np.pi
        expected_f32 = data_f32.astype(np.float64) * two_pi
        expected_i32 = data_i32.astype(np.float64) * two_pi
        assert np.allclose(result_f32[:2], expected_f32, rtol=1e-10)
        assert np.allclose(result_i32[:2], expected_i32, rtol=1e-10)

    def test_numerical_precision(self, n_qubits):
        """Test numerical precision with very small values."""
        # Use very small values
        small_data = np.array([1e-300, 1e-200, 1e-100])
        result = encode(small_data, n_qubits)

        # Should not overflow or underflow
        assert np.all(np.isfinite(result[:3])), "Result should not contain NaN or Inf"

        # Values should be very small (close to zero)
        assert np.all(result[:3] < 1e-99), "Very small inputs should produce very small outputs"

    def test_large_values(self, n_qubits):
        """Test behavior with values that could cause issues."""
        # Note: The function doesn't clip to [0,1], so values > 1 will produce angles > 2π
        large_data = np.array([2.0, 3.0, 5.0])
        result = encode(large_data, n_qubits)

        # Should still work but produce larger angles
        two_pi = 2.0 * np.pi
        check_count = min(len(large_data), n_qubits)
        expected = large_data[:check_count] * two_pi
        assert np.allclose(result[:check_count], expected, rtol=1e-10)

    def test_empty_array(self, n_qubits):
        """Test encoding of empty array."""
        empty_data = np.array([])
        result = encode(empty_data, n_qubits)

        # Should produce array of zeros
        assert result.shape == (n_qubits,), f"Expected shape ({n_qubits},), got {result.shape}"
        assert np.allclose(result, 0.0, atol=1e-10), "Empty input should produce zeros"
