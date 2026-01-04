"""
Integration tests for PennyLane SIMD Angle Encoding plugin.

Tests cover basic functionality, gradient computation, adjoint operations,
and integration with various PennyLane devices.
"""

import pytest
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

try:
    from pennylane_simd_angle import AngleEncoding, AngleEncodingBatch
    from pennylane_simd_angle.angle import (
        angle_encoding_function,
        angle_encoding_batch_function
    )
except ImportError:
    pytest.skip("PennyLane SIMD plugin not available", allow_module_level=True)


class TestAngleEncodingBasic:
    """Test basic angle encoding functionality."""

    def test_basic_encoding(self):
        """Test basic angle encoding in QNode."""
        dev = qml.device('default.qubit', wires=4)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(4))
            return qml.state()

        data = pnp.array([0.0, 0.25, 0.5, 0.75])
        result = circuit(data)

        assert result is not None
        assert isinstance(result, pnp.ndarray)
        assert result.shape == (16,)  # 2^4 statevector

    def test_wires_as_int(self):
        """Test that single wire as integer works."""
        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=0)
            return qml.state()

        data = pnp.array([0.5])
        result = circuit(data)

        assert result is not None
        assert result.shape == (2,)

    def test_data_length_validation(self):
        """Test that data length must match number of wires."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            return qml.state()

        # Should work
        data = pnp.array([0.1, 0.2])
        result = circuit(data)
        assert result is not None

        # Should fail - wrong length
        with pytest.raises(ValueError, match="Data length.*must match"):
            bad_data = pnp.array([0.1, 0.2, 0.3])
            circuit(bad_data)

    def test_data_dimension_validation(self):
        """Test that data must be 1D."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            return qml.state()

        # Should fail - 2D data
        with pytest.raises(ValueError, match="Data must be 1D"):
            bad_data = pnp.array([[0.1, 0.2], [0.3, 0.4]])
            circuit(bad_data)

    def test_encoding_correctness(self):
        """Test that encoding produces correct angles."""
        data = np.array([0.0, 0.5, 1.0])
        expected = data * 2 * np.pi

        # Use functional API
        angles = angle_encoding_function(data, n_qubits=3)

        assert np.allclose(angles, expected)


class TestAngleEncodingGradients:
    """Test gradient computation support."""

    def test_gradient_computation_backprop(self):
        """Test gradient flows through encoding with backprop."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            qml.RY(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        data = pnp.array([0.1, 0.2], requires_grad=True)
        result = circuit(data)
        grad = qml.grad(circuit)(data)

        assert grad is not None
        assert grad.shape == data.shape
        assert isinstance(grad, pnp.ndarray)

    def test_gradient_computation_param_shift(self):
        """Test gradient with parameter shift."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        data = pnp.array([0.1, 0.2], requires_grad=True)
        grad = qml.grad(circuit)(data)

        assert grad is not None
        assert grad.shape == data.shape

    def test_gradient_finite_diff(self):
        """Test gradient with finite differences."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, diff_method="finite-diff")
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        data = pnp.array([0.1, 0.2], requires_grad=True)
        grad = qml.grad(circuit)(data)

        assert grad is not None

    def test_higher_order_gradients(self):
        """Test computation of second-order gradients."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        data = pnp.array([0.1, 0.2], requires_grad=True)

        # First gradient
        grad_fn = qml.grad(circuit)
        grad = grad_fn(data)

        # Second gradient (Hessian diagonal)
        hess_fn = qml.grad(grad_fn)
        hess = hess_fn(data)

        assert hess is not None
        assert hess.shape == data.shape


class TestAngleEncodingAdjoint:
    """Test adjoint (inverse) operation."""

    def test_adjoint_operation(self):
        """Test adjoint (inverse) operation returns to initial state."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            qml.adjoint(AngleEncoding)(data, wires=range(2))
            return qml.state()

        data = pnp.array([0.1, 0.2])
        result = circuit(data)

        # Should return to |00⟩ state
        expected = pnp.array([1.0, 0.0, 0.0, 0.0])
        assert pnp.allclose(result, expected, atol=1e-10)

    def test_adjoint_method(self):
        """Test the adjoint() method."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(data):
            op = AngleEncoding(data, wires=range(2))
            qml.adjoint(op)(data, wires=range(2))
            return qml.state()

        data = pnp.array([0.3, 0.4])
        result = circuit(data)

        expected = pnp.array([1.0, 0.0, 0.0, 0.0])
        assert pnp.allclose(result, expected, atol=1e-10)


class TestAngleEncodingDecomposition:
    """Test operation decomposition."""

    def test_decomposition(self):
        """Test that decomposition produces correct RY gates."""
        data = pnp.array([0.1, 0.2, 0.3])
        op = AngleEncoding(data, wires=range(3))

        decomp = op.decomposition()

        assert len(decomp) == 3
        assert all(isinstance(gate, qml.RY) for gate in decomp)

        # Check angles match
        for i, gate in enumerate(decomp):
            assert pnp.allclose(gate.parameters[0], op.angles[i])
            assert gate.wires == qml.wires.Wires([i])


class TestAngleEncodingBatch:
    """Test batch encoding functionality."""

    def test_batch_encoding_functional(self):
        """Test batch encoding with functional API."""
        batch = np.random.random((10, 4))
        angles = angle_encoding_batch_function(batch, n_qubits=4)

        assert angles.shape == (10, 4)
        assert np.all(angles >= 0) and np.all(angles <= 2 * np.pi)

    def test_batch_encoding_operation(self):
        """Test AngleEncodingBatch operation."""
        batch_data = pnp.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])

        op = AngleEncodingBatch(batch_data, wires=range(2))

        assert op.batch_size == 3
        assert op.angles_batch.shape == (3, 2)

    def test_batch_validation(self):
        """Test batch validation."""
        # Wrong dimension
        with pytest.raises(ValueError, match="must be 2D"):
            bad_data = pnp.array([0.1, 0.2, 0.3])
            AngleEncodingBatch(bad_data, wires=range(2))

        # Wrong size
        with pytest.raises(ValueError, match="must match"):
            bad_data = pnp.array([[0.1, 0.2, 0.3]])
            AngleEncodingBatch(bad_data, wires=range(2))


class TestAngleEncodingIntegration:
    """Test integration with PennyLane workflows."""

    def test_integration_with_cnot(self):
        """Test encoding followed by entanglement."""
        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(3))
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.state()

        data = pnp.array([0.1, 0.2, 0.3])
        result = circuit(data)

        assert result.shape == (8,)

    def test_integration_with_rotations(self):
        """Test encoding with additional rotations."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            qml.RZ(0.3, wires=0)
            qml.RX(0.5, wires=1)
            return qml.expval(qml.PauliZ(0))

        data = pnp.array([0.2, 0.4])
        result = circuit(data)

        assert isinstance(result, (float, pnp.ndarray))

    def test_integration_variational_circuit(self):
        """Test in a variational circuit context."""
        dev = qml.device('default.qubit', wires=4)

        # Encoding layer
        def encoding_layer(data):
            AngleEncoding(data, wires=range(4))

        # Variational layer
        def variational_layer(params):
            for i in range(4):
                qml.RY(params[i], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])

        @qml.qnode(dev, diff_method="backprop")
        def circuit(data, params):
            encoding_layer(data)
            variational_layer(params)
            return qml.expval(qml.PauliZ(0))

        data = pnp.array([0.1, 0.2, 0.3, 0.4])
        params = pnp.array([0.5, 0.6, 0.7, 0.8], requires_grad=True)

        result = circuit(data, params)
        assert result is not None

        # Test gradient
        grad = qml.grad(circuit)(data, params)
        assert grad is not None


class TestAngleEncodingPerformance:
    """Performance and correctness tests."""

    def test_encoding_range(self):
        """Test that encoded angles are in [0, 2π]."""
        data = np.random.random(100)
        angles = angle_encoding_function(data, n_qubits=100)

        assert np.all(angles >= 0)
        assert np.all(angles <= 2 * np.pi)

    def test_encoding_deterministic(self):
        """Test that encoding is deterministic."""
        data = np.array([0.1, 0.2, 0.3, 0.4])

        angles1 = angle_encoding_function(data, n_qubits=4)
        angles2 = angle_encoding_function(data, n_qubits=4)

        assert np.allclose(angles1, angles2)

    def test_encoding_zero_data(self):
        """Test encoding of zero data."""
        data = np.zeros(4)
        angles = angle_encoding_function(data, n_qubits=4)

        expected = np.zeros(4)
        assert np.allclose(angles, expected)

    def test_encoding_ones_data(self):
        """Test encoding of ones data."""
        data = np.ones(4)
        angles = angle_encoding_function(data, n_qubits=4)

        expected = 2 * np.pi * np.ones(4)
        assert np.allclose(angles, expected)


class TestAngleEncodingDevices:
    """Test with different PennyLane devices."""

    def test_default_qubit_device(self):
        """Test with default.qubit device."""
        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(3))
            return qml.state()

        data = pnp.array([0.1, 0.2, 0.3])
        result = circuit(data)
        assert result is not None

    @pytest.mark.skipif(
        not hasattr(qml, 'default_mixed'),
        reason="default.mixed device not available"
    )
    def test_default_mixed_device(self):
        """Test with default.mixed device."""
        dev = qml.device('default.mixed', wires=2)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            return qml.state()

        data = pnp.array([0.1, 0.2])
        result = circuit(data)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
