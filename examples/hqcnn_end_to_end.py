#!/usr/bin/env python3
"""
End-to-End HQCNN Example with Batch Encoding

This example demonstrates a complete HQCNN (High-Performance Quantum Convolutional
Neural Network) implementation using the SIMD-accelerated Pennylane plugin with
batch encoding for efficient training.

Features:
- Batch encoding for efficient data processing
- Complete training pipeline with loss computation
- Gradient computation and optimization
- Evaluation on test data
- Performance comparison: SIMD vs NumPy encoding
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import time
from typing import Tuple, Optional

try:
    from pennylane_simd_angle import AngleEncoding, angle_encoding_batch_function
    print("✓ SIMD Angle Encoding plugin loaded")
except ImportError:
    print("✗ SIMD plugin not available. Please install:")
    print("  pip install simd-angle-encoder")
    exit(1)


# ============================================================================
# Data Generation and Preprocessing
# ============================================================================

def generate_circle_dataset(n_samples: int = 100, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple circle dataset for binary classification.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        X: Feature array of shape (n_samples, 2)
        y: Labels of shape (n_samples,) with values ±1
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random points
    X = np.random.uniform(-1, 1, (n_samples, 2))

    # Label based on whether point is inside unit circle
    y = np.array([1 if np.sqrt(x[0]**2 + x[1]**2) < 0.7 else -1 for x in X])

    return X, y


def normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range.

    Args:
        X: Input data

    Returns:
        Normalized data
    """
    # Min-max normalization to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int,
                   shuffle: bool = True) -> list:
    """Create mini-batches from dataset.

    Args:
        X: Feature array
        y: Labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before batching

    Returns:
        List of (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]

    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    batches = []
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        batches.append((X_batch, y_batch))

    return batches


# ============================================================================
# Quantum Circuit Definitions
# ============================================================================

def create_hqcnn_circuit(n_qubits: int, n_layers: int = 2):
    """Create an HQCNN circuit with angle encoding and variational layers.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        QNode function
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(data, params):
        # Encoding layer using SIMD-accelerated angle encoding
        AngleEncoding(data, wires=range(n_qubits))

        # Variational layers
        for layer in range(n_layers):
            # Rotations
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)

            # Entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit


def create_hqcnn_circuit_with_numpy_encoding(n_qubits: int, n_layers: int = 2):
    """Create HQCNN circuit using NumPy encoding (for comparison).

    This uses standard NumPy angle encoding instead of SIMD-accelerated encoding.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(data, params):
        # Encoding layer using standard NumPy operations
        for i in range(n_qubits):
            qml.RY(data[i] * 2 * np.pi, wires=i)

        # Variational layers (same as SIMD version)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    return circuit


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def predict(circuit, X: np.ndarray, params: pnp.ndarray) -> np.ndarray:
    """Make predictions on a dataset.

    Args:
        circuit: QNode circuit
        X: Feature array
        params: Circuit parameters

    Returns:
        Predictions array
    """
    predictions = np.array([circuit(x, params) for x in X])
    return np.sign(predictions)  # Convert to ±1


def compute_accuracy(circuit, X: np.ndarray, y: np.ndarray,
                    params: pnp.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        circuit: QNode circuit
        X: Feature array
        y: True labels
        params: Circuit parameters

    Returns:
        Accuracy score
    """
    predictions = predict(circuit, X, params)
    accuracy = np.mean(predictions == y)
    return accuracy


def compute_loss(circuit, X: np.ndarray, y: np.ndarray,
                params: pnp.ndarray) -> pnp.ndarray:
    """Compute mean squared error loss.

    Args:
        circuit: QNode circuit
        X: Feature array
        y: True labels
        params: Circuit parameters

    Returns:
        Loss value
    """
    predictions = pnp.array([circuit(x, params) for x in X])
    loss = pnp.mean((predictions - y) ** 2)
    return loss


def train_epoch(circuit, batches: list, params: pnp.ndarray,
                learning_rate: float = 0.1) -> Tuple[pnp.ndarray, float]:
    """Train for one epoch.

    Args:
        circuit: QNode circuit
        batches: List of (X_batch, y_batch) tuples
        params: Circuit parameters
        learning_rate: Learning rate for gradient descent

    Returns:
        Updated parameters and average loss
    """
    total_loss = 0.0
    n_batches = len(batches)

    for X_batch, y_batch in batches:
        # Convert to PennyLane arrays with gradient tracking
        X_batch_pl = pnp.array(X_batch, requires_grad=False)
        y_batch_pl = pnp.array(y_batch, requires_grad=False)

        # Compute loss for batch
        def loss_fn(params):
            predictions = pnp.array([circuit(x, params) for x in X_batch])
            return pnp.mean((predictions - y_batch_pl) ** 2)

        # Compute gradients
        grad = qml.grad(loss_fn)(params)

        # Update parameters using gradient descent
        params = params - learning_rate * grad

        # Track loss
        batch_loss = loss_fn(params)
        total_loss += batch_loss

    avg_loss = total_loss / n_batches
    return params, avg_loss


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_end_to_end_training(
    n_qubits: int = 4,
    n_layers: int = 2,
    n_train_samples: int = 80,
    n_test_samples: int = 20,
    n_epochs: int = 10,
    batch_size: int = 10,
    learning_rate: float = 0.1,
    seed: int = 42
):
    """Run end-to-end HQCNN training pipeline.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        seed: Random seed
    """
    print("\n" + "="*70)
    print("End-to-End HQCNN Training with SIMD Batch Encoding")
    print("="*70)

    # Set random seeds
    np.random.seed(seed)
    pnp.random.seed(seed)

    # 1. Generate and preprocess data
    print("\n[1] Generating dataset...")
    X_train, y_train = generate_circle_dataset(n_train_samples, seed=seed)
    X_test, y_test = generate_circle_dataset(n_test_samples, seed=seed + 1)

    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Test samples: {X_test.shape[0]}")
    print(f"    Feature dimension: {X_train.shape[1]}")

    # Pad data to match n_qubits if needed
    if X_train.shape[1] < n_qubits:
        pad_width = n_qubits - X_train.shape[1]
        X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant')
        print(f"    Padded features to {n_qubits} dimensions")

    # Normalize data
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    # 2. Create batches
    print("\n[2] Creating batches...")
    train_batches = create_batches(X_train, y_train, batch_size, shuffle=True)
    print(f"    Number of batches: {len(train_batches)}")
    print(f"    Batch size: {batch_size}")

    # 3. Initialize circuit and parameters
    print("\n[3] Initializing HQCNN circuit...")
    circuit = create_hqcnn_circuit(n_qubits, n_layers)

    # Initialize parameters randomly
    params_shape = (n_layers, n_qubits, 2)  # (layers, qubits, rotations_per_qubit)
    params = pnp.array(
        np.random.uniform(0, 2 * np.pi, params_shape) * 0.1,
        requires_grad=True
    )
    print(f"    Parameters shape: {params.shape}")
    print(f"    Total parameters: {params.size}")

    # 4. Training loop
    print("\n[4] Training HQCNN...")
    print(f"    Epochs: {n_epochs}")
    print(f"    Learning rate: {learning_rate}")
    print("\n    Epoch | Train Loss | Train Acc | Test Acc | Time (s)")
    print("   " + "-"*55)

    train_times = []

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train one epoch
        params, avg_loss = train_epoch(
            circuit, train_batches, params, learning_rate
        )

        # Compute accuracy
        train_acc = compute_accuracy(circuit, X_train, y_train, params)
        test_acc = compute_accuracy(circuit, X_test, y_test, params)

        epoch_time = time.time() - start_time
        train_times.append(epoch_time)

        print(f"    {epoch+1:5d} |   {avg_loss:.4f}   |  {train_acc:.4f}   | "
              f"{test_acc:.4f}  | {epoch_time:.3f}")

    avg_epoch_time = np.mean(train_times)
    print(f"\n    Average epoch time: {avg_epoch_time:.3f}s")

    # 5. Final evaluation
    print("\n[5] Final Evaluation...")
    train_acc = compute_accuracy(circuit, X_test, y_test, params)
    print(f"    Test accuracy: {train_acc:.4f}")

    # 6. Demonstrate batch encoding performance
    print("\n[6] Batch Encoding Performance...")
    test_batch = X_test[:10]

    # SIMD encoding
    start_time = time.time()
    for _ in range(100):
        angles_simd = angle_encoding_batch_function(test_batch, n_qubits=n_qubits)
    simd_time = time.time() - start_time

    # NumPy encoding
    start_time = time.time()
    for _ in range(100):
        angles_numpy = test_batch * 2 * np.pi
    numpy_time = time.time() - start_time

    speedup = numpy_time / simd_time
    print(f"    SIMD time (100 iterations): {simd_time:.4f}s")
    print(f"    NumPy time (100 iterations): {numpy_time:.4f}s")
    print(f"    Speedup: {speedup:.2f}x")

    # 7. Sample predictions
    print("\n[7] Sample Predictions...")
    print("    Data Point       | True Label | Prediction | Match")
    print("   " + "-"*50)

    n_samples_show = min(5, len(X_test))
    for i in range(n_samples_show):
        pred = circuit(X_test[i], params)
        pred_label = 1 if pred > 0 else -1
        match = "✓" if pred_label == y_test[i] else "✗"
        data_str = f"[{X_test[i, 0]:.2f}, {X_test[i, 1]:.2f}]"
        print(f"    {data_str:14s} |     {y_test[i]:2d}    |     {pred_label:2d}     |  {match}")

    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)

    return params, circuit


# ============================================================================
# Comparison: SIMD vs NumPy Encoding
# ============================================================================

def compare_simd_vs_numpy(n_epochs: int = 5, seed: int = 42):
    """Compare training performance between SIMD and NumPy encoding.

    Args:
        n_epochs: Number of training epochs
        seed: Random seed
    """
    print("\n" + "="*70)
    print("Performance Comparison: SIMD vs NumPy Encoding")
    print("="*70)

    # Generate dataset
    n_qubits = 4
    n_layers = 2
    n_train_samples = 50
    batch_size = 10

    X_train, y_train = generate_circle_dataset(n_train_samples, seed=seed)
    X_train = np.pad(X_train, ((0, 0), (0, n_qubits - X_train.shape[1])), mode='constant')
    X_train = normalize_data(X_train)

    train_batches = create_batches(X_train, y_train, batch_size)

    # Test with SIMD encoding
    print("\n[SIMD] Training with SIMD-accelerated encoding...")
    circuit_simd = create_hqcnn_circuit(n_qubits, n_layers)
    params_simd = pnp.array(
        np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 2)) * 0.1,
        requires_grad=True
    )

    start_time = time.time()
    for epoch in range(n_epochs):
        params_simd, loss = train_epoch(circuit_simd, train_batches, params_simd)
    simd_time = time.time() - start_time

    print(f"    Total time: {simd_time:.3f}s")
    print(f"    Final loss: {loss:.4f}")

    # Test with NumPy encoding
    print("\n[NumPy] Training with standard NumPy encoding...")
    circuit_numpy = create_hqcnn_circuit_with_numpy_encoding(n_qubits, n_layers)
    params_numpy = pnp.array(
        np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 2)) * 0.1,
        requires_grad=True
    )

    start_time = time.time()
    for epoch in range(n_epochs):
        params_numpy, loss = train_epoch(circuit_numpy, train_batches, params_numpy)
    numpy_time = time.time() - start_time

    print(f"    Total time: {numpy_time:.3f}s")
    print(f"    Final loss: {loss:.4f}")

    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print(f"    SIMD encoding: {simd_time:.3f}s")
    print(f"    NumPy encoding: {numpy_time:.3f}s")
    print(f"    Speedup: {numpy_time/simd_time:.2f}x")
    print("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("HQCNN End-to-End Example with Batch Encoding")
    print("="*70)

    # Run complete training pipeline
    params, circuit = run_end_to_end_training(
        n_qubits=4,
        n_layers=2,
        n_train_samples=80,
        n_test_samples=20,
        n_epochs=10,
        batch_size=10,
        learning_rate=0.1,
        seed=42
    )

    # Run performance comparison
    compare_simd_vs_numpy(n_epochs=5, seed=42)

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
