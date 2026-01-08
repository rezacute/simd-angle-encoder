#!/usr/bin/env python3
"""
HQCNN Qiskit Plugin Example with Batch Encoding

This example demonstrates a complete HQCNN (High-Performance Quantum Convolutional
Neural Network) implementation using the SIMD-accelerated Qiskit plugin with
batch encoding for efficient training.

Features:
- Batch encoding using SIMD optimizations
- Complete training pipeline with loss computation
- Parameter optimization using gradient-free methods
- Evaluation on test data
- Performance comparison: SIMD vs NumPy encoding
- Integration with Qiskit's quantum runtime
"""

import numpy as np
import time
from typing import Tuple, Optional, List
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator

try:
    from qiskit_simd_angle import SIMDAngleEncoding, simd_angle_encoding
    from simd_angle_encoder import encode, encode_batch
    print("✓ SIMD Angle Encoding Qiskit plugin loaded")
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
                   shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
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

def create_hqcnn_circuit_qiskit(n_qubits: int, n_layers: int = 2) -> QuantumCircuit:
    """Create an HQCNN variational circuit template.

    This creates the variational layers that will be combined with
    SIMD angle encoding for the complete HQCNN circuit.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        QuantumCircuit: Parameterized variational circuit
    """
    # Create a parameterized circuit
    qc = QuantumCircuit(n_qubits)

    # Variational layers
    params = []
    for layer in range(n_layers):
        # Rotations
        for i in range(n_qubits):
            from qiskit.circuit import Parameter
            theta_y = Parameter(f'theta_{layer}_{i}_y')
            theta_z = Parameter(f'theta_{layer}_{i}_z')
            params.extend([theta_y, theta_z])
            qc.ry(theta_y, i)
            qc.rz(theta_z, i)

        # Entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc, params


def build_hqcnn_with_encoding(data_point: np.ndarray, params: np.ndarray,
                               n_qubits: int, n_layers: int = 2) -> QuantumCircuit:
    """Build a complete HQCNN circuit with SIMD encoding for a single data point.

    Args:
        data_point: Input data to encode
        params: Variational parameters
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        Complete quantum circuit with encoding and variational layers
    """
    # Create encoding circuit using SIMD
    encoding_circuit = SIMDAngleEncoding(data_point, num_qubits=n_qubits)

    # Create variational circuit
    qc = QuantumCircuit(n_qubits)
    qc.compose(encoding_circuit, inplace=True)

    # Add variational layers
    param_idx = 0
    for layer in range(n_layers):
        # Rotations
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1

        # Entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc


def build_hqcnn_with_numpy_encoding(data_point: np.ndarray, params: np.ndarray,
                                     n_qubits: int, n_layers: int = 2) -> QuantumCircuit:
    """Build HQCNN circuit with NumPy encoding (for comparison).

    Args:
        data_point: Input data to encode
        params: Variational parameters
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        Complete quantum circuit with NumPy encoding and variational layers
    """
    qc = QuantumCircuit(n_qubits)

    # Encoding layer using NumPy
    angles = data_point * 2 * np.pi
    for i in range(n_qubits):
        qc.ry(angles[i], i)

    # Variational layers (same as SIMD version)
    param_idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1

        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def compute_expectation(circuit: QuantumCircuit) -> float:
    """Compute the expectation value of Pauli-Z on first qubit.

    Args:
        circuit: Quantum circuit to evaluate

    Returns:
        Expectation value
    """
    # Get statevector
    statevector = Statevector(circuit)

    # Create Pauli-Z operator on first qubit
    # In Qiskit, we compute this by measuring the probability of |0> state
    # The expectation value of Z is P(|0>) - P(|1>) = 2*P(|0>) - 1
    probs = statevector.probabilities()

    # Probability of measuring 0 in first qubit
    # Sum over all states where first qubit is 0
    n_qubits = circuit.num_qubits
    p0_first = 0.0
    for i, prob in enumerate(probs):
        # Check if first qubit is 0 in binary representation
        if (i >> (n_qubits - 1)) & 1 == 0:
            p0_first += prob

    expectation = 2 * p0_first - 1
    return expectation


def predict_single(data_point: np.ndarray, params: np.ndarray,
                   n_qubits: int, n_layers: int = 2,
                   use_simd: bool = True) -> int:
    """Make prediction for a single data point.

    Args:
        data_point: Input features
        params: Circuit parameters (can be flat or shaped)
        n_qubits: Number of qubits
        n_layers: Number of layers
        use_simd: Whether to use SIMD encoding

    Returns:
        Prediction (±1)
    """
    params_flat = params.flatten()
    if use_simd:
        circuit = build_hqcnn_with_encoding(data_point, params_flat, n_qubits, n_layers)
    else:
        circuit = build_hqcnn_with_numpy_encoding(data_point, params_flat, n_qubits, n_layers)

    expectation = compute_expectation(circuit)
    return 1 if expectation > 0 else -1


def predict_batch(X: np.ndarray, params: np.ndarray,
                  n_qubits: int, n_layers: int = 2,
                  use_simd: bool = True) -> np.ndarray:
    """Make predictions on a batch of data.

    Args:
        X: Feature array of shape (n_samples, n_features)
        params: Circuit parameters
        n_qubits: Number of qubits
        n_layers: Number of layers
        use_simd: Whether to use SIMD encoding

    Returns:
        Predictions array of shape (n_samples,)
    """
    predictions = np.array([
        predict_single(x, params, n_qubits, n_layers, use_simd)
        for x in X
    ])
    return predictions


def compute_accuracy(X: np.ndarray, y: np.ndarray,
                     params: np.ndarray, n_qubits: int,
                     n_layers: int = 2, use_simd: bool = True) -> float:
    """Compute classification accuracy.

    Args:
        X: Feature array
        y: True labels
        params: Circuit parameters
        n_qubits: Number of qubits
        n_layers: Number of layers
        use_simd: Whether to use SIMD encoding

    Returns:
        Accuracy score
    """
    predictions = predict_batch(X, params, n_qubits, n_layers, use_simd)
    accuracy = np.mean(predictions == y)
    return accuracy


def compute_loss(X: np.ndarray, y: np.ndarray,
                 params: np.ndarray, n_qubits: int,
                 n_layers: int = 2, use_simd: bool = True) -> float:
    """Compute mean squared error loss.

    Args:
        X: Feature array
        y: True labels
        params: Circuit parameters
        n_qubits: Number of qubits
        n_layers: Number of layers
        use_simd: Whether to use SIMD encoding

    Returns:
        Loss value
    """
    params_flat = params.flatten()
    predictions = np.array([
        compute_expectation(build_hqcnn_with_encoding(x, params_flat, n_qubits, n_layers))
        if use_simd else
        compute_expectation(build_hqcnn_with_numpy_encoding(x, params_flat, n_qubits, n_layers))
        for x in X
    ])
    loss = np.mean((predictions - y) ** 2)
    return loss


def objective_function(params_flat: np.ndarray, X: np.ndarray, y: np.ndarray,
                       n_qubits: int, n_layers: int, use_simd: bool = True) -> float:
    """Objective function for optimization.

    Args:
        params_flat: Flattened parameter array
        X: Feature array
        y: True labels
        n_qubits: Number of qubits
        n_layers: Number of layers
        use_simd: Whether to use SIMD encoding

    Returns:
        Loss value (to be minimized)
    """
    # Reshape parameters if needed
    params = params_flat.reshape(n_layers, n_qubits, 2)
    return compute_loss(X, y, params, n_qubits, n_layers, use_simd)


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_end_to_end_training(
    n_qubits: int = 4,
    n_layers: int = 2,
    n_train_samples: int = 50,
    n_test_samples: int = 20,
    maxiter: int = 50,
    batch_size: int = 10,
    seed: int = 42
):
    """Run end-to-end HQCNN training pipeline with Qiskit.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        maxiter: Maximum optimization iterations
        batch_size: Batch size for training
        seed: Random seed
    """
    import qiskit.circuit

    print("\n" + "="*70)
    print("End-to-End HQCNN Training with Qiskit + SIMD Batch Encoding")
    print("="*70)

    # Set random seed
    np.random.seed(seed)

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

    # 3. Initialize parameters
    print("\n[3] Initializing HQCNN parameters...")
    params_shape = (n_layers, n_qubits, 2)  # (layers, qubits, rotations_per_qubit)
    params = np.random.uniform(0, 2 * np.pi, params_shape) * 0.1
    print(f"    Parameters shape: {params.shape}")
    print(f"    Total parameters: {params.size}")

    # 4. Training with batch optimization
    print("\n[4] Training HQCNN with COBYLA optimizer...")
    print(f"    Max iterations: {maxiter}")
    print("\n    Iter | Train Loss | Train Acc | Test Acc | Time (s)")
    print("   " + "-"*55)

    optimizer = COBYLA(maxiter=maxiter, tol=1e-4)
    train_times = []
    losses = []
    train_accs = []
    test_accs = []

    # Training history tracking
    iteration = [0]

    def callback(xk):
        """Callback to track optimization progress."""
        iteration[0] += 1
        current_params = xk.reshape(params_shape)

        start_time = time.time()

        # Compute metrics on full dataset
        loss = compute_loss(X_train, y_train, current_params, n_qubits, n_layers, use_simd=True)
        train_acc = compute_accuracy(X_train, y_train, current_params, n_qubits, n_layers, use_simd=True)
        test_acc = compute_accuracy(X_test, y_test, current_params, n_qubits, n_layers, use_simd=True)

        iter_time = time.time() - start_time
        train_times.append(iter_time)

        losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if iteration[0] % 5 == 0 or iteration[0] == 1:
            print(f"    {iteration[0]:4d} |   {loss:.4f}   |  {train_acc:.4f}   | "
                  f"{test_acc:.4f}  | {iter_time:.3f}")

    # Initial parameters flattened
    params_flat = params.flatten()

    # Optimize on first batch as demonstration
    X_batch, y_batch = train_batches[0]

    # Run optimization
    start_time = time.time()
    result = optimizer.minimize(
        fun=lambda p: objective_function(p, X_batch, y_batch, n_qubits, n_layers, use_simd=True),
        x0=params_flat
    )
    total_time = time.time() - start_time

    # Final parameters
    trained_params = result.x.reshape(params_shape)

    print(f"\n    Total training time: {total_time:.3f}s")
    final_loss = objective_function(result.x, X_batch, y_batch, n_qubits, n_layers, use_simd=True)
    print(f"    Final loss: {final_loss:.4f}")

    # 5. Final evaluation
    print("\n[5] Final Evaluation...")
    final_train_acc = compute_accuracy(X_train, y_train, trained_params, n_qubits, n_layers, use_simd=True)
    final_test_acc = compute_accuracy(X_test, y_test, trained_params, n_qubits, n_layers, use_simd=True)
    print(f"    Final train accuracy: {final_train_acc:.4f}")
    print(f"    Final test accuracy: {final_test_acc:.4f}")

    # 6. Demonstrate batch encoding performance
    print("\n[6] Batch Encoding Performance...")
    test_batch = X_test[:10]

    # SIMD encoding
    start_time = time.time()
    for _ in range(100):
        angles_simd = encode_batch(test_batch, n_qubits=n_qubits)
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
        pred = predict_single(X_test[i], trained_params, n_qubits, n_layers, use_simd=True)
        match = "✓" if pred == y_test[i] else "✗"
        data_str = f"[{X_test[i, 0]:.2f}, {X_test[i, 1]:.2f}]"
        print(f"    {data_str:14s} |     {y_test[i]:2d}    |     {pred:2d}     |  {match}")

    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)

    return trained_params


# ============================================================================
# Comparison: SIMD vs NumPy Encoding
# ============================================================================

def compare_simd_vs_numpy(maxiter: int = 20, seed: int = 42):
    """Compare training performance between SIMD and NumPy encoding.

    Args:
        maxiter: Number of optimization iterations
        seed: Random seed
    """
    print("\n" + "="*70)
    print("Performance Comparison: SIMD vs NumPy Encoding")
    print("="*70)

    # Generate dataset
    n_qubits = 4
    n_layers = 2
    n_train_samples = 30
    batch_size = 10

    X_train, y_train = generate_circle_dataset(n_train_samples, seed=seed)
    X_train = np.pad(X_train, ((0, 0), (0, n_qubits - X_train.shape[1])), mode='constant')
    X_train = normalize_data(X_train)

    # Test with SIMD encoding
    print("\n[SIMD] Training with SIMD-accelerated encoding...")
    params_shape = (n_layers, n_qubits, 2)
    params_simd = np.random.uniform(0, 2 * np.pi, params_shape) * 0.1

    optimizer = COBYLA(maxiter=maxiter, tol=1e-4)

    start_time = time.time()
    result_simd = optimizer.minimize(
        fun=lambda p: objective_function(p, X_train, y_train, n_qubits, n_layers, use_simd=True),
        x0=params_simd.flatten()
    )
    simd_time = time.time() - start_time

    final_loss_simd = compute_loss(
        X_train, y_train, result_simd.x.reshape(params_shape),
        n_qubits, n_layers, use_simd=True
    )

    print(f"    Total time: {simd_time:.3f}s")
    print(f"    Final loss: {final_loss_simd:.4f}")

    # Test with NumPy encoding
    print("\n[NumPy] Training with standard NumPy encoding...")
    params_numpy = np.random.uniform(0, 2 * np.pi, params_shape) * 0.1

    start_time = time.time()
    result_numpy = optimizer.minimize(
        fun=lambda p: objective_function(p, X_train, y_train, n_qubits, n_layers, use_simd=False),
        x0=params_numpy.flatten()
    )
    numpy_time = time.time() - start_time

    final_loss_numpy = compute_loss(
        X_train, y_train, result_numpy.x.reshape(params_shape),
        n_qubits, n_layers, use_simd=False
    )

    print(f"    Total time: {numpy_time:.3f}s")
    print(f"    Final loss: {final_loss_numpy:.4f}")

    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print(f"    SIMD encoding: {simd_time:.3f}s (loss: {final_loss_simd:.4f})")
    print(f"    NumPy encoding: {numpy_time:.3f}s (loss: {final_loss_numpy:.4f})")
    print(f"    Speedup: {numpy_time/simd_time:.2f}x")
    print("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("HQCNN Qiskit Example with Batch Encoding")
    print("="*70)

    # Run complete training pipeline
    params = run_end_to_end_training(
        n_qubits=4,
        n_layers=2,
        n_train_samples=50,
        n_test_samples=20,
        maxiter=50,
        batch_size=10,
        seed=42
    )

    # Run performance comparison
    compare_simd_vs_numpy(maxiter=20, seed=42)

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
