"""
Property-based tests for angle encoding.

Uses Hypothesis to test invariants and properties that should always hold true
regardless of the input values.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from simd_angle_encoder import encode, encode_batch


@pytest.mark.property
class TestEncodingProperties:
    """Test suite for property-based tests of encoding functionality."""

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_output_range_property(self, data):
        """Property: All encoded values should be in [0, 2π]."""
        data_array = np.array(data, dtype=np.float64)
        n_qubits = min(len(data_array), 100)

        result = encode(data_array, n_qubits)

        # All values should be in [0, 2π]
        assert np.all(result >= 0.0), "Encoded values should be >= 0"
        assert np.all(result <= 2.0 * np.pi), "Encoded values should be <= 2π"

    @given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_linearity_property(self, data):
        """Property: Encoding should be linear (encoding(ax) = a*encoding(x))."""
        data_array = np.array(data, dtype=np.float64)
        n_qubits = min(len(data_array), 50)

        # Encode original data
        result_original = encode(data_array, n_qubits)

        # Scale data by 2.0
        scaled_data = data_array * 2.0
        result_scaled = encode(scaled_data, n_qubits)

        # Check linearity for first elements (where data exists)
        two_pi = 2.0 * np.pi
        check_count = min(len(data_array), n_qubits)

        # encoding(2*x) should equal 2*encoding(x)
        expected_scaled = result_original[:check_count] * 2.0
        assert np.allclose(result_scaled[:check_count], expected_scaled, rtol=1e-10), \
            "Encoding should be linear"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_zero_encoding_property(self, data):
        """Property: Encoding zero should give zero."""
        data_array = np.array(data, dtype=np.float64)
        n_qubits = min(len(data_array), 30)

        # Create zero array of same size
        zero_array = np.zeros_like(data_array)

        result_data = encode(data_array, n_qubits)
        result_zero = encode(zero_array, n_qubits)

        # Zero should encode to zero
        check_count = min(len(data_array), n_qubits)
        assert np.allclose(result_zero[:check_count], 0.0, atol=1e-10), \
            "Zero input should encode to zero"

        # Regular data should not be all zeros
        assert not np.allclose(result_data[:check_count], 0.0, atol=1e-10) or \
               np.allclose(data_array[:check_count], 0.0, atol=1e-10), \
            "Non-zero data should produce non-zero output (or be close to zero)"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_determinism_property(self, data):
        """Property: Multiple encodings of same data should be identical."""
        data_array = np.array(data, dtype=np.float64)
        n_qubits = min(len(data_array), 50)

        # Encode multiple times
        result1 = encode(data_array, n_qubits)
        result2 = encode(data_array, n_qubits)
        result3 = encode(data_array, n_qubits)

        # All should be identical
        assert np.allclose(result1, result2, rtol=1e-10), \
            "Multiple encodings should be identical (1 vs 2)"
        assert np.allclose(result2, result3, rtol=1e-10), \
            "Multiple encodings should be identical (2 vs 3)"
        assert np.allclose(result1, result3, rtol=1e-10), \
            "Multiple encodings should be identical (1 vs 3)"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_monotonicity_property(self, data):
        """Property: If x1 < x2, then encoding(x1) < encoding(x2)."""
        data_array = np.array(data, dtype=np.float64)
        n_qubits = min(len(data_array), 30)

        result = encode(data_array, n_qubits)

        # Check monotonicity for encoded elements
        two_pi = 2.0 * np.pi
        check_count = min(len(data_array), n_qubits)

        for i in range(check_count - 1):
            if data_array[i] < data_array[i + 1]:
                assert result[i] < result[i + 1], \
                    f"Encoding should preserve order: {data_array[i]} < {data_array[i+1]}"
            elif data_array[i] > data_array[i + 1]:
                assert result[i] > result[i + 1], \
                    f"Encoding should preserve order: {data_array[i]} > {data_array[i+1]}"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_output_shape_property(self, data):
        """Property: Output shape should always be (n_qubits,)."""
        data_array = np.array(data, dtype=np.float64)

        # Test various n_qubits
        for n_qubits in [5, 10, 20, 50, 100]:
            result = encode(data_array, n_qubits)

            assert result.shape == (n_qubits,), \
                f"Output shape should be ({n_qubits},), got {result.shape}"
            assert result.dtype == np.float64, \
                f"Output dtype should be float64, got {result.dtype}"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=10, max_size=100))
    @settings(max_examples=50)
    def test_padding_property(self, data):
        """Property: When n_qubits > data size, extra elements should be zero."""
        data_array = np.array(data, dtype=np.float64)
        data_len = len(data_array)

        # Use n_qubits larger than data size
        n_qubits = data_len + 20

        result = encode(data_array, n_qubits)

        # First data_len elements should be encoded
        two_pi = 2.0 * np.pi
        expected_first = data_array * two_pi
        assert np.allclose(result[:data_len], expected_first, rtol=1e-10), \
            "First elements should be encoded correctly"

        # Padding should be zeros
        assert np.allclose(result[data_len:], 0.0, atol=1e-10), \
            "Padding should be zeros"

    @given(st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                 min_size=1, max_size=50),
        min_size=1, max_size=20
    ))
    @settings(max_examples=50)
    def test_batch_output_shape_property(self, batch_data):
        """Property: Batch output shape should be (batch_size, n_qubits)."""
        batch_array = np.array(batch_data, dtype=np.float64)
        batch_size = batch_array.shape[0]

        # Test various n_qubits
        for n_qubits in [5, 10, 20]:
            result = encode_batch(batch_array, n_qubits)

            assert result.shape == (batch_size, n_qubits), \
                f"Output shape should be ({batch_size}, {n_qubits}), got {result.shape}"
            assert result.dtype == np.float64, \
                f"Output dtype should be float64, got {result.dtype}"

    @given(st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                 min_size=1, max_size=30),
        min_size=2, max_size=10
    ))
    @settings(max_examples=50)
    def test_batch_range_property(self, batch_data):
        """Property: All batch-encoded values should be in [0, 2π]."""
        batch_array = np.array(batch_data, dtype=np.float64)
        n_qubits = 20

        result = encode_batch(batch_array, n_qubits)

        # All values should be in [0, 2π]
        assert np.all(result >= 0.0), "All encoded values should be >= 0"
        assert np.all(result <= 2.0 * np.pi), "All encoded values should be <= 2π"

    @given(st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                 min_size=5, max_size=20),
        min_size=1, max_size=10
    ))
    @settings(max_examples=50)
    def test_batch_consistency_with_single(self, batch_data):
        """Property: Batch encoding should be consistent with single encoding."""
        from simd_angle_encoder import encode

        batch_array = np.array(batch_data, dtype=np.float64)
        n_qubits = 15

        # Encode as batch
        batch_result = encode_batch(batch_array, n_qubits)

        # Encode each row individually
        for i, row in enumerate(batch_array):
            single_result = encode(row, n_qubits)

            # Should match corresponding batch row
            assert np.allclose(single_result, batch_result[i], rtol=1e-10), \
                f"Row {i}: Batch and single encoding should match"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_finite_values_property(self, data):
        """Property: Encoding finite inputs should produce finite outputs."""
        data_array = np.array(data, dtype=np.float64)
        n_qubits = min(len(data_array), 50)

        result = encode(data_array, n_qubits)

        # All values should be finite (no NaN or Inf)
        assert np.all(np.isfinite(result)), \
            "Encoding finite inputs should produce finite outputs"

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_size_invariance(self, size):
        """Property: Encoding should work for any size."""
        data = np.random.random(size)
        n_qubits = min(size, 50)

        result = encode(data, n_qubits)

        assert result.shape == (n_qubits,)
        assert np.all(np.isfinite(result))

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=2, max_size=50),
           st.integers(min_value=0, max_value=49))
    @settings(max_examples=50)
    def test_index_access_property(self, data, index):
        """Property: Result indexing should work correctly."""
        data_array = np.array(data, dtype=np.float64)
        n_qubits = min(len(data_array), 50)

        # Ensure index is in bounds
        if index >= n_qubits:
            index = n_qubits - 1

        result = encode(data_array, n_qubits)

        # Should be able to index
        value = result[index]
        assert isinstance(value, (float, np.floating))
        assert np.isfinite(value)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=10, max_size=100))
    @settings(max_examples=50)
    def test_truncation_property(self, data):
        """Property: When data size > n_qubits, should truncate to n_qubits."""
        data_array = np.array(data, dtype=np.float64)
        data_len = len(data_array)

        # Use n_qubits smaller than data size
        n_qubits = min(data_len // 2, 50)

        result = encode(data_array, n_qubits)

        # Should have correct shape
        assert result.shape == (n_qubits,), f"Expected shape ({n_qubits},), got {result.shape}"

        # First n_qubits should be encoded correctly
        two_pi = 2.0 * np.pi
        expected = data_array[:n_qubits] * two_pi
        assert np.allclose(result, expected, rtol=1e-10), \
            "First n_qubits elements should be encoded correctly"

    @given(st.lists(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                 min_size=1, max_size=20),
        min_size=1, max_size=10
    ))
    @settings(max_examples=50)
    def test_batch_row_independence_property(self, batch_data):
        """Property: Each row in batch should be independent."""
        batch_array = np.array(batch_data, dtype=np.float64)
        n_qubits = 15

        result = encode_batch(batch_array, n_qubits)

        # Each row should have correct encoding
        two_pi = 2.0 * np.pi
        for i, row in enumerate(batch_array):
            check_count = min(len(row), n_qubits)
            expected = row[:check_count] * two_pi
            assert np.allclose(result[i, :check_count], expected, rtol=1e-10), \
                f"Row {i} encoding incorrect"

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_single_element_encoding_property(self, value):
        """Property: Single element should encode correctly."""
        data = np.array([value])
        n_qubits = 10

        result = encode(data, n_qubits)

        # First element should be value * 2π
        expected_first = value * 2.0 * np.pi
        assert np.allclose(result[0], expected_first, rtol=1e-10), \
            f"Single element encoding incorrect for value {value}"

        # Rest should be zeros
        assert np.allclose(result[1:], 0.0, atol=1e-10), \
            "Padding should be zeros"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=10, max_size=50))
    @settings(max_examples=50)
    def test_array_unchanged_property(self, data):
        """Property: Input array should not be modified."""
        data_array = np.array(data, dtype=np.float64)
        original = data_array.copy()

        n_qubits = min(len(data_array), 30)
        result = encode(data_array, n_qubits)

        # Input should be unchanged
        assert np.array_equal(data_array, original), \
            "Input array should not be modified"

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=100),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_repeated_qubit_counts_property(self, data, n_qubits):
        """Property: Encoding should work for various qubit counts."""
        data_array = np.array(data, dtype=np.float64)

        result = encode(data_array, n_qubits)

        assert result.shape == (n_qubits,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 2.0 * np.pi)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_numpy_conversion_property(self, data):
        """Property: Lists should be converted to numpy arrays correctly."""
        n_qubits = min(len(data), 30)

        # Encode list
        result_list = encode(data, n_qubits)

        # Encode numpy array
        data_array = np.array(data, dtype=np.float64)
        result_array = encode(data_array, n_qubits)

        # Should be identical
        assert np.allclose(result_list, result_array, rtol=1e-10), \
            "List and array input should produce identical results"
