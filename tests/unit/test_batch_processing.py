"""
Unit tests for batch encoding functionality.

Tests the batch processing functions with various inputs and edge cases.
"""

import pytest
import numpy as np
from simd_angle_encoder import encode_batch


@pytest.mark.unit
class TestBatchProcessing:
    """Test suite for batch encoding functionality."""

    def test_basic_batch_encoding(self, small_batch, n_qubits):
        """Test basic batch encoding."""
        result = encode_batch(small_batch, n_qubits)

        # Check output shape
        batch_size = small_batch.shape[0]
        assert result.shape == (batch_size, n_qubits), \
            f"Expected shape ({batch_size}, {n_qubits}), got {result.shape}"

        # Check output dtype
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"

        # Check output range [0, 2π]
        assert np.all(result >= 0.0), "All encoded values should be >= 0"
        assert np.all(result <= 2.0 * np.pi), "All encoded values should be <= 2π"

    def test_batch_size_variations(self, batch_size, n_qubits):
        """Test with different batch sizes."""
        data_dim = 16
        batch_data = np.random.random((batch_size, data_dim))

        result = encode_batch(batch_data, n_qubits)

        assert result.shape == (batch_size, n_qubits), \
            f"Expected shape ({batch_size}, {n_qubits}), got {result.shape}"

    def test_single_row_batch(self, n_qubits):
        """Test batch encoding with a single row."""
        single_row = np.array([[0.25, 0.5, 0.75]])
        result = encode_batch(single_row, n_qubits)

        assert result.shape == (1, n_qubits), f"Expected shape (1, {n_qubits}), got {result.shape}"

        # Check first row encoding
        two_pi = 2.0 * np.pi
        check_count = min(3, n_qubits)
        expected = single_row[0, :check_count] * two_pi
        assert np.allclose(result[0, :check_count], expected, rtol=1e-10)

    def test_batch_known_values(self, batch_known_data):
        """Test batch encoding with known deterministic values."""
        n_qubits = 10
        result = encode_batch(batch_known_data, n_qubits)

        two_pi = 2.0 * np.pi
        batch_size = batch_known_data.shape[0]

        # Check each row
        for i in range(batch_size):
            check_count = min(batch_known_data.shape[1], n_qubits)
            expected = batch_known_data[i, :check_count] * two_pi
            assert np.allclose(result[i, :check_count], expected, rtol=1e-10), \
                f"Row {i} encoding mismatch"

    def test_batch_zeros(self):
        """Test batch encoding of zeros."""
        batch_zeros = np.zeros((5, 8))
        n_qubits = 10
        result = encode_batch(batch_zeros, n_qubits)

        # All should be zeros
        assert np.allclose(result, 0.0, atol=1e-10), "Zero batch should produce zero output"

    def test_batch_ones(self):
        """Test batch encoding of ones."""
        batch_ones = np.ones((5, 8))
        n_qubits = 10
        result = encode_batch(batch_ones, n_qubits)

        two_pi = 2.0 * np.pi
        check_count = min(8, n_qubits)

        # All should be 2π
        assert np.allclose(result[:, :check_count], two_pi, rtol=1e-10), \
            "Ones batch should produce 2π output"

    def test_1d_input_reshaped(self, n_qubits):
        """Test that 1D input is properly reshaped to batch."""
        data_1d = np.array([0.25, 0.5, 0.75, 1.0])
        result = encode_batch(data_1d, n_qubits)

        # Should be reshaped to (1, data_dim)
        assert result.shape == (1, n_qubits), f"Expected shape (1, {n_qubits}), got {result.shape}"

        # Check encoding
        two_pi = 2.0 * np.pi
        check_count = min(len(data_1d), n_qubits)
        expected = data_1d[:check_count] * two_pi
        assert np.allclose(result[0, :check_count], expected, rtol=1e-10)

    def test_different_data_dims(self, n_qubits):
        """Test with different data dimensions."""
        batch_size = 5
        data_dims = [4, 8, 16, 32, 64]

        for data_dim in data_dims:
            batch_data = np.random.random((batch_size, data_dim))
            result = encode_batch(batch_data, n_qubits)

            assert result.shape == (batch_size, n_qubits), \
                f"Expected shape ({batch_size}, {n_qubits}) for data_dim={data_dim}, got {result.shape}"

    def test_list_input_batch(self, n_qubits):
        """Test that list inputs work for batch."""
        batch_list = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = encode_batch(batch_list, n_qubits)

        batch_size = len(batch_list)
        assert result.shape == (batch_size, n_qubits), \
            f"Expected shape ({batch_size}, {n_qubits}), got {result.shape}"

    def test_mixed_dtypes_batch(self, n_qubits):
        """Test batch with mixed dtypes."""
        batch_f32 = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
        batch_i32 = np.array([[1, 0], [0, 1]], dtype=np.int32)

        result_f32 = encode_batch(batch_f32, n_qubits)
        result_i32 = encode_batch(batch_i32, n_qubits)

        # Both should work
        assert result_f32.dtype == np.float64
        assert result_i32.dtype == np.float64

    def test_data_dim_larger_than_n_qubits(self):
        """Test when data dimension is larger than n_qubits."""
        data_dim = 20
        n_qubits = 10
        batch_size = 3

        batch_data = np.random.random((batch_size, data_dim))
        result = encode_batch(batch_data, n_qubits)

        # Should truncate to n_qubits
        assert result.shape == (batch_size, n_qubits)

        # Check first n_qubits are encoded correctly
        two_pi = 2.0 * np.pi
        for i in range(batch_size):
            expected = batch_data[i, :n_qubits] * two_pi
            assert np.allclose(result[i], expected, rtol=1e-10), \
                f"Row {i} encoding mismatch"

    def test_data_dim_smaller_than_n_qubits(self):
        """Test when data dimension is smaller than n_qubits."""
        data_dim = 5
        n_qubits = 10
        batch_size = 3

        batch_data = np.random.random((batch_size, data_dim))
        result = encode_batch(batch_data, n_qubits)

        # Should pad with zeros
        assert result.shape == (batch_size, n_qubits)

        # Check first data_dim elements
        two_pi = 2.0 * np.pi
        for i in range(batch_size):
            expected_full = np.zeros(n_qubits)
            expected_full[:data_dim] = batch_data[i] * two_pi
            assert np.allclose(result[i], expected_full, rtol=1e-10), \
                f"Row {i} encoding mismatch"

    def test_batch_consistency(self, n_qubits):
        """Test that batch encoding is consistent with individual encoding."""
        from simd_angle_encoder import encode

        data = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

        # Encode individually
        individual_result = encode(data, n_qubits)

        # Encode as batch
        batch_data = data.reshape(1, -1)
        batch_result = encode_batch(batch_data, n_qubits)

        # Should be identical
        assert np.allclose(individual_result, batch_result[0], rtol=1e-10), \
            "Individual and batch encoding should produce identical results"

    def test_large_batch(self, n_qubits):
        """Test with larger batch size."""
        batch_size = 100
        data_dim = 32

        batch_data = np.random.random((batch_size, data_dim))
        result = encode_batch(batch_data, n_qubits)

        assert result.shape == (batch_size, n_qubits)

        # Check all values are in valid range
        assert np.all(result >= 0.0)
        assert np.all(result <= 2.0 * np.pi)

    def test_batch_numerical_stability(self):
        """Test numerical stability with extreme values."""
        batch_size = 2
        n_qubits = 5

        # Very small values
        small_batch = np.array([[1e-300, 1e-200, 0.5], [1e-150, 1e-100, 0.5]])
        result = encode_batch(small_batch, n_qubits)

        # Should not overflow/underflow
        assert np.all(np.isfinite(result))

    def test_3d_input_raises_error(self, n_qubits):
        """Test that 3D input raises appropriate error."""
        data_3d = np.array([[[0.1, 0.2], [0.3, 0.4]]])

        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            encode_batch(data_3d, n_qubits)
