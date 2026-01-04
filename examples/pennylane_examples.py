"""
Example: Basic PennyLane SIMD Angle Encoding Usage

This example demonstrates the basic usage of the SIMD-accelerated angle
encoding plugin for PennyLane.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

# Import the SIMD encoding operation
try:
    from pennylane_simd_angle import AngleEncoding
    print("✓ SIMD Angle Encoding plugin loaded successfully")
except ImportError:
    print("✗ SIMD plugin not available. Please install:")
    print("  pip install simd-angle-encoder")
    exit(1)


def example_basic_usage():
    """Example 1: Basic circuit with angle encoding."""
    print("\n" + "="*60)
    print("Example 1: Basic Angle Encoding")
    print("="*60)

    # Create device
    dev = qml.device('default.qubit', wires=4)

    # Define circuit
    @qml.qnode(dev)
    def circuit(data):
        AngleEncoding(data, wires=range(4))
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        return qml.expval(qml.PauliZ(0))

    # Execute circuit
    data = [0.1, 0.2, 0.3, 0.4]
    result = circuit(data)

    print(f"Input data: {data}")
    print(f"Circuit result: {result:.4f}")
    print(f"Expected value of Pauli-Z on qubit 0: {result:.4f}")


def example_gradient_computation():
    """Example 2: Computing gradients with backpropagation."""
    print("\n" + "="*60)
    print("Example 2: Gradient Computation")
    print("="*60)

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(data):
        AngleEncoding(data, wires=range(2))
        qml.RY(0.5, wires=0)
        return qml.expval(qml.PauliZ(0))

    # Data with gradient tracking
    data = pnp.array([0.1, 0.2], requires_grad=True)

    # Forward pass
    result = circuit(data)
    print(f"Circuit output: {result:.4f}")

    # Compute gradient
    grad = qml.grad(circuit)(data)
    print(f"Gradient: {grad}")

    # Verify gradient numerically
    epsilon = 1e-7
    numerical_grad = pnp.zeros_like(data)
    for i in range(len(data)):
        data_plus = data.copy()
        data_plus[i] += epsilon
        result_plus = circuit(data_plus)

        data_minus = data.copy()
        data_minus[i] -= epsilon
        result_minus = circuit(data_minus)

        numerical_grad[i] = (result_plus - result_minus) / (2 * epsilon)

    print(f"Numerical gradient: {numerical_grad}")
    print(f"Gradient match: {pnp.allclose(grad, numerical_grad, atol=1e-5)}")


def example_adjoint_operation():
    """Example 3: Using adjoint (inverse) operations."""
    print("\n" + "="*60)
    print("Example 3: Adjoint Operation")
    print("="*60)

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit_with_adjoint(data):
        AngleEncoding(data, wires=range(2))
        qml.adjoint(AngleEncoding)(data, wires=range(2))
        return qml.state()

    data = pnp.array([0.5, 0.7])
    result = circuit_with_adjoint(data)

    print(f"Input data: {data}")
    print(f"State after encoding + adjoint(encoding):")
    print(f"  {result}")
    print(f"Should be |00⟩ state: [1, 0, 0, 0]")
    print(f"Match: {pnp.allclose(result, [1, 0, 0, 0], atol=1e-10)}")


def example_quantum_neural_network():
    """Example 4: Simple quantum neural network."""
    print("\n" + "="*60)
    print("Example 4: Quantum Neural Network")
    print("="*60)

    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev, diff_method="backprop")
    def qnn_circuit(data, params):
        # Encoding layer
        AngleEncoding(data, wires=range(4))

        # Variational layer 1
        for i in range(4):
            qml.RY(params[i], wires=i)

        # Entanglement
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])

        # Variational layer 2
        for i in range(4):
            qml.RZ(params[i + 4], wires=i)

        return qml.expval(qml.PauliZ(0))

    # Initialize parameters
    params = pnp.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        requires_grad=True
    )

    # Input data
    data = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

    # Forward pass
    output = qnn_circuit(data, params)
    print(f"QNN output: {output:.4f}")

    # Compute gradients
    data_grad = qml.grad(qnn_circuit, argnum=0)(data, params)
    param_grad = qml.grad(qnn_circuit, argnum=1)(data, params)

    print(f"Gradient w.r.t. data: {data_grad}")
    print(f"Gradient w.r.t. params: {param_grad}")


def example_batch_processing():
    """Example 5: Batch processing for multiple samples."""
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)

    from pennylane_simd_angle.angle import angle_encoding_batch_function

    # Create batch of data
    batch_size = 10
    n_qubits = 4
    batch_data = np.random.random((batch_size, n_qubits))

    print(f"Batch shape: {batch_data.shape}")

    # Encode entire batch at once
    angles_batch = angle_encoding_batch_function(batch_data, n_qubits=n_qubits)

    print(f"Encoded batch shape: {angles_batch.shape}")
    print(f"Sample angles (first row): {angles_batch[0]}")

    # Use in circuits
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def process_sample(angles):
        for i in range(n_qubits):
            qml.RY(angles[i], wires=i)
        return qml.expval(qml.PauliZ(0))

    # Process all samples
    results = [process_sample(angles) for angles in angles_batch]

    print(f"Number of results: {len(results)}")
    print(f"Sample results: {[f'{r:.4f}' for r in results[:3]]}")


def example_variational_classification():
    """Example 6: Simple variational classifier."""
    print("\n" + "="*60)
    print("Example 6: Variational Quantum Classifier")
    print("="*60)

    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev, diff_method="backprop")
    def classifier_circuit(data, params):
        # Encoding
        AngleEncoding(data, wires=range(4))

        # Variational layers
        for layer in range(2):
            for i in range(4):
                qml.RY(params[layer * 8 + i], wires=i)
            for i in range(3):
                qml.CNOT(wires=[i, i+1])

        return qml.expval(qml.PauliZ(0))

    # Training data
    X_train = pnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ])

    y_train = pnp.array([1.0, -1.0, 1.0])

    # Initialize parameters
    np.random.seed(42)
    params = pnp.array(
        np.random.random(16) * 0.1,
        requires_grad=True
    )

    # Loss function
    def loss(params, X, y):
        predictions = pnp.array([classifier_circuit(x, params) for x in X])
        return pnp.mean((predictions - y) ** 2)

    # Training loop
    print("Training variational classifier...")
    initial_loss = loss(params, X_train, y_train)
    print(f"Initial loss: {initial_loss:.4f}")

    for epoch in range(5):
        # Compute gradients
        grad = qml.grad(loss)(params, X_train, y_train)

        # Gradient descent update
        params = params - 0.1 * grad

        current_loss = loss(params, X_train, y_train)
        print(f"Epoch {epoch + 1}, Loss: {current_loss:.4f}")

    # Test prediction
    test_data = pnp.array([0.3, 0.4, 0.5, 0.6])
    prediction = classifier_circuit(test_data, params)
    print(f"\nPrediction for {test_data}: {prediction:.4f}")
    print(f"Class: {1 if prediction > 0 else -1}")


def example_multiple_devices():
    """Example 7: Using with different PennyLane devices."""
    print("\n" + "="*60)
    print("Example 7: Multiple Device Support")
    print("="*60)

    data = [0.1, 0.2, 0.3, 0.4]

    # Test with default.qubit
    dev_default = qml.device('default.qubit', wires=4)

    @qml.qnode(dev_default)
    def circuit_default(data):
        AngleEncoding(data, wires=range(4))
        return qml.state()

    result_default = circuit_default(data)
    print(f"default.qubit result: {result_default.shape} statevector")

    # Test with lightning.qubit if available
    try:
        dev_lightning = qml.device('lightning.qubit', wires=4)

        @qml.qnode(dev_lightning)
        def circuit_lightning(data):
            AngleEncoding(data, wires=range(4))
            return qml.state()

        result_lightning = circuit_lightning(data)
        print(f"lightning.qubit result: {result_lightning.shape} statevector")

        # Verify they match
        match = np.allclose(result_default, result_lightning)
        print(f"Results match: {match}")
    except Exception as e:
        print(f"lightning.qubit not available: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PennyLane SIMD Angle Encoder - Examples")
    print("="*60)

    example_basic_usage()
    example_gradient_computation()
    example_adjoint_operation()
    example_quantum_neural_network()
    example_batch_processing()
    example_variational_classification()
    example_multiple_devices()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
