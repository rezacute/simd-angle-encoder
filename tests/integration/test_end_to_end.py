"""
End-to-end integration tests.

Tests complete workflows from data input to encoded output.
"""

import pytest
import numpy as np
from simd_angle_encoder import encode, encode_batch, benchmark, simd_info


@pytest.mark.integration
class TestEndToEnd:
    """Test suite for end-to-end workflows."""

    def test_complete_encoding_workflow(self):
        """Test a complete encoding workflow from start to finish."""
        # Simulate realistic data preprocessing
        raw_data = np.random.random(100)

        # Normalize to [0, 1] if needed (already in range)
        data = raw_data

        # Encode
        n_qubits = 50
        encoded = encode(data, n_qubits)

        # Verify the workflow
        assert encoded.shape == (n_qubits,)
        assert np.all(encoded >= 0.0)
        assert np.all(encoded <= 2.0 * np.pi)
        assert np.all(np.isfinite(encoded))

    def test_batch_preprocessing_workflow(self):
        """Test complete batch workflow with preprocessing."""
        # Simulate a dataset
        batch_size = 20
        data_dim = 32

        # Generate random data
        raw_batch = np.random.random((batch_size, data_dim))

        # Optional: normalize each row to [0, 1]
        batch_data = raw_batch

        # Encode batch
        n_qubits = 25
        encoded_batch = encode_batch(batch_data, n_qubits)

        # Verify
        assert encoded_batch.shape == (batch_size, n_qubits)
        assert np.all(encoded_batch >= 0.0)
        assert np.all(encoded_batch <= 2.0 * np.pi)
        assert np.all(np.isfinite(encoded_batch))

    def test_quantum_circuit_preparation_workflow(self):
        """Test workflow for preparing data for quantum circuits."""
        # Simulate classical data preparation for quantum ML
        n_samples = 10
        n_features = 16
        n_qubits = 12

        # Generate feature data
        features = np.random.random((n_samples, n_features))

        # Encode to angles for quantum circuit
        angles = encode_batch(features, n_qubits)

        # Verify angles are valid for quantum circuits
        assert angles.shape == (n_samples, n_qubits)

        # All angles should be in [0, 2π]
        assert np.all(angles >= 0.0)
        assert np.all(angles <= 2.0 * np.pi)

        # Check that angles are reasonable (not all zeros or identical)
        assert not np.allclose(angles, 0.0, atol=1e-10)
        assert not np.all(np.abs(angles - angles[0]) < 1e-10)

    def test_data_normalization_workflow(self):
        """Test workflow with data normalization."""
        # Simulate data in range [-5, 5]
        raw_data = np.random.uniform(-5, 5, 50)

        # Normalize to [0, 1]
        data_min, data_max = raw_data.min(), raw_data.max()
        normalized_data = (raw_data - data_min) / (data_max - data_min)

        # Encode
        n_qubits = 30
        encoded = encode(normalized_data, n_qubits)

        # Verify
        assert encoded.shape == (n_qubits,)
        assert np.all(encoded >= 0.0)
        assert np.all(encoded <= 2.0 * np.pi)

    def test_multi_scale_data_workflow(self):
        """Test workflow with data at different scales."""
        # Data with different magnitudes
        small_scale = np.random.random(10) * 0.01
        medium_scale = np.random.random(10) * 1.0
        large_scale = np.random.random(10) * 100.0

        # Normalize each scale
        small_norm = (small_scale - small_scale.min()) / (small_scale.max() - small_scale.min())
        medium_norm = (medium_scale - medium_scale.min()) / (medium_scale.max() - medium_scale.min())
        large_norm = (large_scale - large_scale.min()) / (large_scale.max() - large_scale.min())

        # Combine
        combined_data = np.concatenate([small_norm, medium_norm, large_norm])

        # Encode
        n_qubits = 25
        encoded = encode(combined_data, n_qubits)

        # Verify
        assert encoded.shape == (n_qubits,)
        assert np.all(np.isfinite(encoded))

    def test_batch_consistency_across_calls(self):
        """Test that batch encoding is consistent across multiple calls."""
        # Generate data
        batch_data = np.random.random((15, 20))
        n_qubits = 18

        # Encode multiple times
        results = []
        for _ in range(5):
            result = encode_batch(batch_data, n_qubits)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], rtol=1e-10), \
                "Multiple calls should produce identical results"

    def test_realistic_dataset_workflow(self):
        """Test workflow with a realistic dataset size."""
        # Simulate a dataset for quantum machine learning
        n_train_samples = 100
        n_test_samples = 20
        n_features = 64
        n_qubits = 32

        # Training data
        train_data = np.random.random((n_train_samples, n_features))
        train_encoded = encode_batch(train_data, n_qubits)

        # Test data
        test_data = np.random.random((n_test_samples, n_features))
        test_encoded = encode_batch(test_data, n_qubits)

        # Verify shapes
        assert train_encoded.shape == (n_train_samples, n_qubits)
        assert test_encoded.shape == (n_test_samples, n_qubits)

        # Verify ranges
        assert np.all(train_encoded >= 0.0) and np.all(train_encoded <= 2.0 * np.pi)
        assert np.all(test_encoded >= 0.0) and np.all(test_encoded <= 2.0 * np.pi)

    def test_benchmark_integration(self):
        """Test that benchmark function works end-to-end."""
        data_size = 128
        batch_size = 10
        n_qubits = 64

        # Run benchmark
        numpy_time, simd_time = benchmark(data_size, batch_size, n_qubits, n_runs=3)

        # Verify times are reasonable
        assert numpy_time > 0, "NumPy time should be positive"
        assert simd_time > 0, "SIMD time should be positive"
        assert isinstance(numpy_time, float)
        assert isinstance(simd_time, float)

        # SIMD should generally be faster or comparable
        # (This might not always be true for small data sizes)
        print(f"NumPy time: {numpy_time:.3f}ms, SIMD time: {simd_time:.3f}ms")

    def test_simd_info_integration(self):
        """Test that SIMD info is accessible."""
        info = simd_info()

        assert isinstance(info, str)
        assert len(info) > 0

        # Should contain some useful information
        assert any(keyword in info for keyword in ["SIMD", "System", "CPU", "Core", "OS", "Architecture"])

    def test_gradient_simulation_workflow(self):
        """Test workflow that might be used in gradient-based optimization."""
        # Simulate parameters that need to be encoded
        params = np.random.random(20)
        n_qubits = 15

        # Encode parameters
        encoded_params = encode(params, n_qubits)

        # Simulate small perturbation (like gradient step)
        epsilon = 1e-6
        perturbed_params = params + epsilon

        # Encode perturbed parameters
        encoded_perturbed = encode(perturbed_params, n_qubits)

        # Check that encoding is smooth (small input change -> small output change)
        # Note: This checks for smoothness, not gradient computation
        difference = np.abs(encoded_perturbed - encoded_params)
        two_pi_epsilon = 2.0 * np.pi * epsilon

        # The change should be related to epsilon
        # (At least for the first elements where params exist)
        check_count = min(len(params), n_qubits)
        assert np.allclose(difference[:check_count], two_pi_epsilon, atol=1e-5, rtol=0.1), \
            "Encoding should be smooth"

    def test_different_qubit_counts_workflow(self):
        """Test workflow with different qubit counts."""
        data = np.random.random(30)

        qubit_counts = [5, 10, 15, 20, 25, 30]

        results = {}
        for n_qubits in qubit_counts:
            encoded = encode(data, n_qubits)
            results[n_qubits] = encoded

            # Verify each
            assert encoded.shape == (n_qubits,)
            assert np.all(encoded >= 0.0)
            assert np.all(encoded <= 2.0 * np.pi)

        # Check that for same input data, first elements are consistent
        # regardless of n_qubits
        reference = results[qubit_counts[0]]
        for n_qubits in qubit_counts[1:]:
            current = results[n_qubits]
            # First elements should match
            assert np.allclose(reference[:min(qubit_counts[0], n_qubits)],
                             current[:min(qubit_counts[0], n_qubits)],
                             rtol=1e-10)

    def test_reproducibility_workflow(self):
        """Test that results are reproducible with same seed."""
        # Set seed
        np.random.seed(12345)

        # Generate and encode
        data1 = np.random.random(50)
        encoded1 = encode(data1, 30)

        # Reset seed and generate again
        np.random.seed(12345)
        data2 = np.random.random(50)
        encoded2 = encode(data2, 30)

        # Should be identical
        assert np.allclose(encoded1, encoded2, rtol=1e-10), \
            "Results should be reproducible with same seed"

    def test_error_recovery_workflow(self):
        """Test workflow with error handling and recovery."""
        # Try with invalid input
        invalid_data = "not an array"

        try:
            # This should raise an error or convert gracefully
            result = encode(invalid_data, 10)
            # If it doesn't raise, result should still be valid
            assert isinstance(result, np.ndarray)
        except (ValueError, TypeError, AttributeError):
            # Expected to raise error
            pass

        # Verify we can continue with valid data
        valid_data = np.random.random(20)
        result = encode(valid_data, 15)
        assert result.shape == (15,)

    @pytest.mark.slow
    def test_large_scale_workflow(self):
        """Test workflow with large-scale data."""
        # Large dataset
        batch_size = 1000
        data_dim = 500
        n_qubits = 200

        batch_data = np.random.random((batch_size, data_dim))
        encoded = encode_batch(batch_data, n_qubits)

        # Verify
        assert encoded.shape == (batch_size, n_qubits)
        assert np.all(np.isfinite(encoded))
        assert np.all(encoded >= 0.0)
        assert np.all(encoded <= 2.0 * np.pi)
