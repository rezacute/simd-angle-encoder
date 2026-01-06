# HQCNN End-to-End Example

Complete implementation of a High-Performance Quantum Convolutional Neural Network (HQCNN) using SIMD-accelerated batch encoding with the PennyLane plugin.

## Overview

This example demonstrates a practical quantum machine learning workflow with:

- **Batch Encoding**: Efficient processing of multiple data samples using SIMD optimization
- **Training Pipeline**: Complete training loop with loss computation and gradient optimization
- **Evaluation**: Accuracy metrics on test data
- **Performance Comparison**: SIMD vs NumPy encoding benchmark

## Features

### 1. Data Generation and Preprocessing
- Synthetic dataset generation (circle classification)
- Data normalization
- Mini-batch creation for efficient training

### 2. Quantum Circuit Architecture
- **Encoding Layer**: SIMD-accelerated angle encoding
- **Variational Layers**: Alternating RY/RZ rotations with CNOT entanglement
- **Measurement**: Pauli-Z expectation value

### 3. Training Components
- Mean squared error (MSE) loss function
- Gradient computation via PennyLane backpropagation
- Parameter updates using gradient descent
- Batch-based training for efficiency

### 4. Performance Optimization
- Batch encoding with SIMD acceleration (30-40x speedup)
- Efficient memory usage with zero-copy operations
- Comparison with NumPy-based encoding

## Usage

### Basic Usage

```bash
cd examples
python hqcnn_end_to_end.py
```

### Expected Output

```
======================================================================
End-to-End HQCNN Training with SIMD Batch Encoding
======================================================================

[1] Generating dataset...
    Training samples: 80
    Test samples: 20
    Feature dimension: 2
    Padded features to 4 dimensions

[2] Creating batches...
    Number of batches: 8
    Batch size: 10

[3] Initializing HQCNN circuit...
    Parameters shape: (2, 4, 2)
    Total parameters: 16

[4] Training HQCNN...
    Epochs: 10
    Learning rate: 0.1

    Epoch | Train Loss | Train Acc | Test Acc | Time (s)
   -------------------------------------------------------
        1 |   0.8923   |  0.5500   | 0.6000  | 0.234
        2 |   0.7851   |  0.6125   | 0.6500  | 0.221
        ...
       10 |   0.3421   |  0.8500   | 0.8000  | 0.218

    Average epoch time: 0.222s

[5] Final Evaluation...
    Test accuracy: 0.8000

[6] Batch Encoding Performance...
    SIMD time (100 iterations): 0.0234s
    NumPy time (100 iterations): 0.7812s
    Speedup: 33.37x

[7] Sample Predictions...
    Data Point       | True Label | Prediction | Match
   --------------------------------------------------
    [0.32, 0.45]    |      1     |      1     |  ✓
    [0.67, 0.78]    |     -1     |     -1     |  ✓
    ...
```

## Code Structure

### Key Functions

#### Data Processing
- `generate_circle_dataset()`: Generate synthetic binary classification data
- `normalize_data()`: Normalize features to [0, 1] range
- `create_batches()`: Create mini-batches from dataset

#### Circuit Definition
- `create_hqcnn_circuit()`: Create quantum circuit with SIMD encoding
- `create_hqcnn_circuit_with_numpy_encoding()`: Create circuit with NumPy encoding (for comparison)

#### Training Functions
- `train_epoch()`: Train for one epoch using batch processing
- `compute_loss()`: Calculate MSE loss
- `compute_accuracy()`: Calculate classification accuracy
- `predict()`: Make predictions on data

#### Main Pipeline
- `run_end_to_end_training()`: Complete training pipeline
- `compare_simd_vs_numpy()`: Performance comparison

## Customization

### Modify Network Architecture

```python
# Change number of qubits and layers
params, circuit = run_end_to_end_training(
    n_qubits=8,      # More qubits for higher capacity
    n_layers=3,      # Deeper network
    ...
)
```

### Adjust Training Hyperparameters

```python
params, circuit = run_end_to_end_training(
    n_epochs=20,           # More training epochs
    batch_size=20,         # Larger batches
    learning_rate=0.05,    # Lower learning rate
    ...
)
```

### Use Different Datasets

Replace the data generation with your own dataset:

```python
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Preprocess
X = normalize_data(X)
y = np.where(y > 0, 1, -1)  # Convert to binary labels

# Train
params, circuit = run_end_to_end_training(
    n_qubits=4,
    X_train=X_train,
    y_train=y_train,
    ...
)
```

## Performance Considerations

### SIMD Batch Encoding

The batch encoding function processes multiple samples in parallel:

```python
# Efficient batch encoding
from pennylane_simd_angle import angle_encoding_batch_function

# Encode entire batch at once
angles = angle_encoding_batch_function(X_batch, n_qubits=4)

# Process through circuits
results = [circuit(angles[i], params) for i in range(len(angles))]
```

**Performance Benefits**:
- 30-40x speedup over NumPy encoding
- Efficient memory usage
- Scalable to large batches

### Training Optimization

1. **Batch Size**: Larger batches (10-20) provide better SIMD utilization
2. **Gradient Computation**: Use `diff_method="backprop"` for speed
3. **Device Choice**: `default.qubit` for simulation, hardware for production

## Circuit Visualization

The HQCNN circuit structure:

```
Data → [Angle Encoding] → [Variational Layer 1] → ... → [Measurement]
                              └─ RY/RZ rotations
                              └─ CNOT entanglement
```

### Variational Layer Detail

```
For each qubit i:
    RY(params[layer, i, 0]) ──●── RZ(params[layer, i, 1])
                               │
For next qubit i+1:           ×
    RY(params[layer, i+1, 0]) ──●── RZ(params[layer, i+1, 1])
```

## Integration with Existing Code

### Use as a Module

```python
from hqcnn_end_to_end import (
    create_hqcnn_circuit,
    train_epoch,
    compute_accuracy,
    create_batches,
    normalize_data
)

# Create circuit
circuit = create_hqcnn_circuit(n_qubits=4, n_layers=2)

# Prepare data
X_normalized = normalize_data(X)
batches = create_batches(X_normalized, y, batch_size=10)

# Train
params = initialize_parameters()
for epoch in range(n_epochs):
    params, loss = train_epoch(circuit, batches, params, learning_rate=0.1)
    print(f"Epoch {epoch}: loss = {loss:.4f}")
```

## Troubleshooting

### Import Errors

```python
# If you see: "ModuleNotFoundError: No module named 'pennylane_simd_angle'"
# Install the package:
pip install simd-angle-encoder

# Or add to PYTHONPATH if developing:
export PYTHONPATH=/path/to/simd-angle-encoder/python:$PYTHONPATH
```

### Out of Memory

For large datasets:
- Reduce batch size
- Use fewer qubits
- Process data in chunks

### Slow Training

- Ensure SIMD plugin is loaded (check for "✓ SIMD Angle Encoding plugin loaded")
- Use `diff_method="backprop"` for faster gradients
- Reduce dataset size for testing

## Advanced Usage

### Custom Loss Functions

```python
def hinge_loss(circuit, X, y, params):
    """Hinge loss for binary classification."""
    predictions = pnp.array([circuit(x, params) for x in X])
    return pnp.mean(pnp.maximum(0, 1 - y * predictions))
```

### Multi-Class Classification

```python
# Modify circuit for multi-class
@qml.qnode(dev)
def multi_class_circuit(data, params):
    AngleEncoding(data, wires=range(n_qubits))

    # Variational layers
    # ...

    # Measure multiple qubits for multi-class output
    return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]
```

## References

- [PennyLane Documentation](https://pennylane.ai/)
- [SIMD Angle Encoder Core](../python/simd_angle_encoder/)
- [PennyLane Plugin](../python/pennylane_simd_angle/)
- [Performance Benchmarks](../benchmarks/pennylane_benchmark.py)

## License

This example is part of the HQCNN SIMD Angle Encoder project.
