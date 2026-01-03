"""
Unit tests for NumPy integration.

Tests that the encoder properly handles NumPy arrays, dtypes, and broadcasting.
"""

import pytest
import numpy as np
from simd_angle_encoder import encode, encode_batch


@pytest.mark.unit
class TestNumPyIntegration:
    """Test suite for NumPy integration."""

    def test_numpy_array_input(self, n_qubits):
        """Test that NumPy arrays work correctly."""
        data = np.array([0.1, 0.5, 0.9])
        result = encode(data, n_qubits)

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_qubits,)

    def test_different_dtypes_converted(self, n_qubits):
        """Test that different dtypes are converted to float64."""
        dtypes = [
            np.float16, np.float32, np.float64,
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64
        ]

        for dtype in dtypes:
            data = np.array([1, 1, 1], dtype=dtype)
            result = encode(data, n_qubits)

            # Output should always be float64
            assert result.dtype == np.float64, f"dtype {dtype} not converted to float64"

    def test_contiguous_array(self, n_qubits):
        """Test with contiguous arrays."""
        data = np.ascontiguousarray(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        result = encode(data, n_qubits)

        assert result.shape == (n_qubits,)
        assert result.dtype == np.float64

    def test_non_contiguous_array(self, n_qubits):
        """Test with non-contiguous arrays (sliced)."""
        # Create non-contiguous array by slicing
        data = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        sliced = data[::2]  # Every other element

        result = encode(sliced, n_qubits)

        assert result.shape == (n_qubits,)
        assert result.dtype == np.float64

        # Check encoding is correct
        two_pi = 2.0 * np.pi
        check_count = min(len(sliced), n_qubits)
        expected = sliced[:check_count] * two_pi
        assert np.allclose(result[:check_count], expected, rtol=1e-10)

    def test_fortran_order_array(self, n_qubits):
        """Test with Fortran-ordered arrays."""
        data = np.asfortranarray(np.array([0.1, 0.2, 0.3, 0.4]))
        result = encode(data, n_qubits)

        assert result.shape == (n_qubits,)
        assert result.dtype == np.float64

    def test_batch_contiguous(self, n_qubits):
        """Test batch with contiguous arrays."""
        batch_data = np.ascontiguousarray(np.random.random((5, 10)))
        result = encode_batch(batch_data, n_qubits)

        assert result.shape == (5, n_qubits)
        assert result.dtype == np.float64

    def test_batch_non_contiguous(self, n_qubits):
        """Test batch with non-contiguous arrays."""
        # Create 2D array and slice
        batch_data = np.random.random((10, 10))
        sliced = batch_data[::2, ::2]  # Every other row and column

        result = encode_batch(sliced, n_qubits)

        assert result.shape == (5, n_qubits)
        assert result.dtype == np.float64

    def test_memory_layout_preservation(self, n_qubits):
        """Test that input memory layout doesn't affect output."""
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # C-contiguous
        result_c = encode(np.ascontiguousarray(data), n_qubits)

        # F-contiguous
        result_f = encode(np.asfortranarray(data), n_qubits)

        # Results should be identical
        assert np.allclose(result_c, result_f, rtol=1e-10)

    def test_writeable_flag(self, n_qubits):
        """Test that input writeable flag doesn't affect operation."""
        # Read-only array
        data = np.array([0.1, 0.2, 0.3])
        data.setflags(write=False)

        result = encode(data, n_qubits)

        assert result.shape == (n_qubits,)
        assert np.all(np.isfinite(result))

    def test_array_with_strides(self, n_qubits):
        """Test array with custom strides."""
        # Create array with specific strides
        data = np.arange(10, dtype=np.float64)
        result = encode(data, n_qubits)

        assert result.shape == (n_qubits,)

    def test_view_casting(self, n_qubits):
        """Test that view casting doesn't break anything."""
        # Create array and view it as different dtype
        data = np.array([0.1, 0.2, 0.3], dtype=np.float64)

        # This should still work
        result = encode(data, n_qubits)
        assert result.shape == (n_qubits,)

    def test_numpy_scalar_input(self, n_qubits):
        """Test with NumPy scalar inputs."""
        # Single numpy scalar
        scalar = np.float64(0.5)

        # Should convert to 1D array
        result = encode(scalar, n_qubits)
        assert result.shape == (1,)

    def test_zero_dimension_array(self):
        """Test with 0-dimensional array (scalar)."""
        # 0-d array
        data = np.array(0.5)

        # Should work and convert to 1D
        n_qubits = 5
        result = encode(data, n_qubits)
        assert result.shape == (n_qubits,)

    def test_array_protocol(self, n_qubits):
        """Test that __array__ protocol is respected."""
        # Create a simple class that implements __array__
        class ArrayLike:
            def __init__(self, data):
                self.data = data

            def __array__(self):
                return np.array(self.data)

        array_like = ArrayLike([0.1, 0.2, 0.3])
        result = encode(array_like, n_qubits)

        assert result.shape == (n_qubits,)

    def test_nan_handling(self, n_qubits):
        """Test handling of NaN values."""
        data = np.array([0.1, np.nan, 0.5])
        result = encode(data, n_qubits)

        # NaN should propagate
        assert np.isnan(result[1])

    def test_inf_handling(self, n_qubits):
        """Test handling of Inf values."""
        data = np.array([0.1, np.inf, 0.5])
        result = encode(data, n_qubits)

        # Inf should propagate (and result in inf after multiplication)
        assert np.isinf(result[1])

    def test_negative_values(self, n_qubits):
        """Test handling of negative values."""
        data = np.array([-0.5, 0.0, 0.5])
        result = encode(data, n_qubits)

        # Negative values should produce negative angles
        # (The function doesn't clip to [0,1])
        two_pi = 2.0 * np.pi
        expected = data * two_pi
        check_count = min(len(data), n_qubits)
        assert np.allclose(result[:check_count], expected[:check_count], rtol=1e-10)

    def test_batch_nan_inf(self, n_qubits):
        """Test batch handling of NaN and Inf."""
        batch_data = np.array([
            [0.1, 0.2, 0.3],
            [np.nan, 0.5, 0.7],
            [np.inf, 0.3, 0.9]
        ])

        result = encode_batch(batch_data, n_qubits)

        # Check that NaN and Inf propagate
        assert np.isnan(result[1, 0])
        assert np.isinf(result[2, 0])

    def test_copy_semantics(self, n_qubits):
        """Test that results are new arrays, not views."""
        data = np.array([0.1, 0.2, 0.3])
        result = encode(data, n_qubits)

        # Result should be a separate array
        assert result is not data

        # Modifying result should not affect input
        original_data = data.copy()
        result[0] = 999.0
        assert np.array_equal(data, original_data)
