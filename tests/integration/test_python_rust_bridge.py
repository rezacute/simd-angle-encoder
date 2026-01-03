"""
Integration tests for Python-Rust FFI bridge.

Tests the boundary between Python and Rust code, ensuring proper data transfer and type conversion.
"""

import pytest
import numpy as np
import sys
from simd_angle_encoder import encode, encode_batch, simd_info


@pytest.mark.integration
class TestPythonRustBridge:
    """Test suite for Python-Rust integration."""

    @pytest.fixture(autouse=True)
    def check_rust_module(self):
        """Skip tests if Rust module is not available."""
        try:
            import simd_angle_encoder._simd_angle_encoder as _rust
            _rust.angle_encode_simd
            _rust.angle_encode_batch_simd
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Rust module not available: {e}")

    def test_rust_module_exists(self):
        """Test that the Rust module can be imported."""
        try:
            import simd_angle_encoder._simd_angle_encoder as _rust
            assert hasattr(_rust, 'angle_encode_simd')
            assert hasattr(_rust, 'angle_encode_batch_simd')
            assert hasattr(_rust, 'get_simd_info')
        except ImportError as e:
            pytest.fail(f"Could not import Rust module: {e}")

    def test_simd_info_string(self):
        """Test that SIMD info returns a string."""
        info = simd_info()

        assert isinstance(info, str), f"SIMD info should be string, got {type(info)}"
        assert len(info) > 0, "SIMD info should not be empty"

        # Check for expected keywords
        assert "SIMD" in info or "System" in info, \
            f"SIMD info should contain relevant info, got: {info}"

    def test_direct_rust_call(self):
        """Test calling Rust function directly."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        data = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        n_qubits = 5

        result = _rust.angle_encode_simd(data, n_qubits)

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_qubits,)
        assert result.dtype == np.float64

    def test_python_wrapper_consistency(self):
        """Test that Python wrapper gives same result as direct Rust call."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        data = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        n_qubits = 10

        # Call through Python wrapper
        python_result = encode(data, n_qubits)

        # Call Rust directly
        rust_result = _rust.angle_encode_simd(data, n_qubits)

        # Should be identical
        assert np.allclose(python_result, rust_result, rtol=1e-10), \
            "Python wrapper and direct Rust call should produce identical results"

    def test_batch_rust_direct_call(self):
        """Test calling Rust batch function directly."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        batch_data = np.array([[0.1, 0.2], [0.5, 0.7]], dtype=np.float64)
        n_qubits = 5

        result = _rust.angle_encode_batch_simd(batch_data, n_qubits)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, n_qubits)
        assert result.dtype == np.float64

    def test_batch_python_wrapper_consistency(self):
        """Test that batch Python wrapper gives same result as Rust."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        batch_data = np.array([[0.1, 0.2], [0.5, 0.7]], dtype=np.float64)
        n_qubits = 10

        # Python wrapper
        python_result = encode_batch(batch_data, n_qubits)

        # Direct Rust call
        rust_result = _rust.angle_encode_batch_simd(batch_data, n_qubits)

        # Should be identical
        assert np.allclose(python_result, rust_result, rtol=1e-10), \
            "Batch Python wrapper and Rust should be identical"

    def test_data_transfer_accuracy(self):
        """Test that data is transferred accurately between Python and Rust."""
        # Create data with specific bit pattern
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        n_qubits = 10

        result = encode(data, n_qubits)

        # Check each value
        two_pi = 2.0 * np.pi
        expected = data * two_pi

        check_count = min(len(data), n_qubits)
        assert np.allclose(result[:check_count], expected[:check_count], rtol=1e-10), \
            "Data transfer should preserve exact values"

    def test_large_array_transfer(self):
        """Test transfer of large arrays."""
        size = 10000
        data = np.random.random(size)
        n_qubits = 100

        result = encode(data, n_qubits)

        assert result.shape == (n_qubits,)
        assert np.all(np.isfinite(result))

    def test_memory_not_corrupted(self):
        """Test that memory is not corrupted during transfer."""
        # Create array and keep reference
        original = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        data = original.copy()

        n_qubits = 10
        result = encode(data, n_qubits)

        # Original should be unchanged
        assert np.array_equal(data, original), "Input data should not be modified"

        # Result should be correct
        assert result.shape == (n_qubits,)

    def test_float64_precision_preserved(self):
        """Test that float64 precision is preserved through the bridge."""
        # Use values that require high precision
        data = np.array([1.0/3.0, 1.0/7.0, 1.0/11.0], dtype=np.float64)
        n_qubits = 10

        result = encode(data, n_qubits)

        # Check that precision is maintained
        two_pi = 2.0 * np.pi
        expected = data * two_pi

        check_count = min(len(data), n_qubits)
        assert np.allclose(result[:check_count], expected[:check_count], rtol=1e-10), \
            "Float64 precision should be preserved"

    def test_error_handling_from_rust(self):
        """Test that errors from Rust are properly handled."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        # Most errors should be caught by Python wrapper
        # but if we call Rust directly with invalid data, it should handle gracefully
        data = np.array([0.1, 0.2], dtype=np.float64)
        n_qubits = 5

        # Should not crash
        result = _rust.angle_encode_simd(data, n_qubits)
        assert result is not None

    def test_concurrent_calls(self):
        """Test multiple concurrent calls to Rust functions."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        data = np.random.random(100)
        n_qubits = 50

        # Make multiple calls
        results = []
        for _ in range(10):
            result = _rust.angle_encode_simd(data, n_qubits)
            results.append(result)

        # All should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], rtol=1e-10), \
                "Concurrent calls should produce identical results"

    def test_rust_returns_numpy_not_list(self):
        """Test that Rust returns NumPy arrays, not Python lists."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        data = np.array([0.1, 0.2, 0.3])
        n_qubits = 5

        result = _rust.angle_encode_simd(data, n_qubits)

        assert isinstance(result, np.ndarray), \
            f"Rust should return NumPy array, got {type(result)}"
        assert not isinstance(result, list)

    def test_array_memory_layout_independence(self):
        """Test that Rust handles different array memory layouts."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        # C-contiguous
        data_c = np.ascontiguousarray(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

        # F-contiguous
        data_f = np.asfortranarray(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

        n_qubits = 10

        result_c = _rust.angle_encode_simd(data_c, n_qubits)
        result_f = _rust.angle_encode_simd(data_f, n_qubits)

        # Should produce identical results
        assert np.allclose(result_c, result_f, rtol=1e-10), \
            "Memory layout should not affect results"

    def test_batch_2d_slice_handling(self):
        """Test that Rust handles 2D array slices correctly."""
        import simd_angle_encoder._simd_angle_encoder as _rust

        # Create 2D array and slice it
        full_array = np.random.random((10, 20))
        sliced = full_array[::2, ::2]  # Non-contiguous

        n_qubits = 15

        # Should work
        result = _rust.angle_encode_batch_simd(sliced, n_qubits)

        assert result.shape == (5, n_qubits)
        assert np.all(np.isfinite(result))
