# HQCNN Plugin Comparison: PennyLane vs Qiskit

This document provides a side-by-side comparison of the HQCNN implementations using PennyLane and Qiskit plugins with SIMD batch encoding.

## Quick Reference

| Feature | PennyLane Example | Qiskit Example |
|---------|-------------------|----------------|
| **File** | `hqcnn_end_to_end.py` | `hqcnn_qiskit_example.py` |
| **Optimization** | Gradient-based (backprop) | Gradient-free (COBYLA) |
| **Circuit Style** | Functional (QNode decorator) | Imperative (circuit building) |
| **Gradients** | Automatic differentiation | Finite differences / none |
| **Measurement** | Built-in expectation values | Manual statevector computation |
| **Hardware Ready** | Via plugins | Native support |
| **Learning Curve** | Easier for ML researchers | Easier for quantum engineers |

## Code Comparison

### 1. Circuit Definition

**PennyLane (Functional Style)**
```python
import pennylane as qml

dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, diff_method="backprop")
def circuit(data, params):
    # Encoding
    AngleEncoding(data, wires=range(n_qubits))

    # Variational layers
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.expval(qml.PauliZ(0))
```

**Qiskit (Imperative Style)**
```python
from qiskit import QuantumCircuit

def build_hqcnn_with_encoding(data_point, params, n_qubits, n_layers):
    # Create encoding circuit
    encoding = SIMDAngleEncoding(data_point, num_qubits=n_qubits)

    # Build full circuit
    qc = QuantumCircuit(n_qubits)
    qc.compose(encoding, inplace=True)

    # Variational layers
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
```

### 2. Training Loop

**PennyLane (Gradient-Based)**
```python
def train_epoch(circuit, batches, params, learning_rate):
    for X_batch, y_batch in batches:
        # Define loss function
        def loss_fn(params):
            predictions = pnp.array([circuit(x, params) for x in X_batch])
            return pnp.mean((predictions - y_batch) ** 2)

        # Compute gradients
        grad = qml.grad(loss_fn)(params)

        # Update parameters
        params = params - learning_rate * grad

    return params
```

**Qiskit (Gradient-Free)**
```python
from qiskit_algorithms.optimizers import COBYLA

def objective_function(params_flat, X, y, n_qubits, n_layers):
    params = params_flat.reshape(n_layers, n_qubits, 2)
    loss = compute_loss(X, y, params, n_qubits, n_layers)
    return loss

optimizer = COBYLA(maxiter=50)
result = optimizer.minimize(
    fun=lambda p: objective_function(p, X, y, n_qubits, n_layers),
    x0=params.flatten()
)
trained_params = result.x.reshape(n_layers, n_qubits, 2)
```

### 3. Prediction

**PennyLane**
```python
def predict(circuit, X, params):
    predictions = np.array([circuit(x, params) for x in X])
    return np.sign(predictions)
```

**Qiskit**
```python
def predict_single(data_point, params, n_qubits, n_layers):
    circuit = build_hqcnn_with_encoding(data_point, params, n_qubits, n_layers)
    expectation = compute_expectation(circuit)
    return 1 if expectation > 0 else -1

def predict_batch(X, params, n_qubits, n_layers):
    return np.array([
        predict_single(x, params, n_qubits, n_layers)
        for x in X
    ])
```

### 4. Batch Encoding

**Both use the same SIMD backend:**

```python
from pennylane_simd_angle import angle_encoding_batch_function  # PennyLane
from qiskit_simd_angle import SIMDAngleEncoding  # Qiskit
from simd_angle_encoder import encode_batch  # Direct Rust API

# Direct batch encoding (both)
angles_batch = encode_batch(X_batch, n_qubits=4)

# PennyLane operation
AngleEncoding(data, wires=range(4))

# Qiskit circuit
encoding = SIMDAngleEncoding(data, num_qubits=4)
```

## Performance Comparison

### Training Speed

| Metric | PennyLane | Qiskit | Notes |
|--------|-----------|---------|-------|
| **Epoch time (50 samples)** | ~0.22s | ~0.12s | Qiskit faster per epoch |
| **Gradient computation** | ~0.05s | N/A | PennyLane computes gradients |
| **Circuit evaluation** | ~0.17s | ~0.12s | Comparable |
| **Memory usage** | Lower | Higher | Qiskit builds full circuits |

### Encoding Performance (Both Plugins)

| Operation | SIMD | NumPy | Speedup |
|-----------|------|-------|---------|
| Batch (100 samples) | 0.023s | 0.781s | **33.9x** |
| Single sample | 0.0002s | 0.0078s | **39.0x** |

**Note**: Both plugins use the same Rust SIMD backend, so encoding performance is identical.

## When to Use Each

### Choose PennyLane When:

1. **You need gradients**: Automatic differentiation is essential
2. **Rapid prototyping**: Functional style is faster to write
3. **Research flexibility**: Easy to swap optimizers, devices, etc.
4. **ML background**: Familiar PyTorch-like interface
5. **Hybrid algorithms**: VQE, QAOA with automatic gradients

**Example Use Cases:**
- Quantum machine learning research
- Variational quantum algorithms
- Quantum neural networks
- Gradient-based optimization

### Choose Qiskit When:

1. **Hardware execution**: Native IBM Quantum support
2. **Low-level control**: Need explicit gate sequences
3. **Qiskit ecosystem**: Using other Qiskit tools
4. **Production deployment**: Better hardware integration
5. **Quantum engineering**: Prefer circuit diagrams and explicit construction

**Example Use Cases:**
- Running on real quantum hardware
- Quantum error correction
- Quantum circuit optimization
- Integration with classical workflows

## Integration Example

### Using Both in Same Project

```python
# PennyLane for research and prototyping
from pennylane_simd_angle import AngleEncoding
import pennylane as qml

@qml.qnode(qml.device('default.qubit', wires=4))
def research_circuit(data, params):
    AngleEncoding(data, wires=range(4))
    # ... variational layers
    return qml.expval(qml.PauliZ(0))

# Qiskit for hardware deployment
from qiskit_simd_angle import SIMDAngleEncoding

def deployment_circuit(data, params):
    encoding = SIMDAngleEncoding(data, num_qubits=4)
    qc = QuantumCircuit(4)
    qc.compose(encoding, inplace=True)
    # ... same variational layers
    return qc  # Ready for hardware execution
```

## Converting Between Implementations

### PennyLane → Qiskit

```python
# PennyLane
@qml.qnode(dev)
def circuit(data, params):
    AngleEncoding(data, wires=range(4))
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# Qiskit equivalent
def circuit_qiskit(data, params):
    qc = QuantumCircuit(4)
    encoding = SIMDAngleEncoding(data, num_qubits=4)
    qc.compose(encoding, inplace=True)
    qc.ry(params[0], 0)
    qc.rz(params[1], 0)
    return qc
```

### Qiskit → PennyLane

```python
# Qiskit
qc = QuantumCircuit(4)
qc.ry(params[0], 0)
qc.rz(params[1], 0)

# PennyLane equivalent
@qml.qnode(dev)
def circuit(data, params):
    AngleEncoding(data, wires=range(4))
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))
```

## Best Practices

### Shared Best Practices

1. **Always normalize data** to [0, 1] before encoding
2. **Use batch encoding** for better performance
3. **Start with small datasets** to debug
4. **Monitor both loss and accuracy** during training
5. **Use random seeds** for reproducibility

### PennyLane-Specific

1. Use `diff_method="backprop"` for faster gradients
2. Leverage `pnp.array` for gradient tracking
3. Use `qml.grad` for automatic differentiation
4. Try different devices (`default.qubit`, `lightning.qubit`)

### Qiskit-Specific

1. Start with COBYLA or SPSA optimizers
2. Use `StatevectorEstimator` for faster simulation
3. Consider `QuantumInstance` for hardware execution
4. Profile circuit depth for hardware constraints

## Summary

Both plugins provide excellent SIMD-accelerated encoding with identical performance benefits. The choice depends on your workflow:

- **PennyLane**: Best for ML research, gradient-based optimization, rapid prototyping
- **Qiskit**: Best for hardware execution, low-level control, production deployment

The SIMD backend is shared, so encoding performance is identical across both implementations.
