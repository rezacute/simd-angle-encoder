# SIMD Angle Encoder - Core API Documentation

**Version**: 0.1.0
**Last Updated**: 2026-01-04

## Overview

The SIMD Angle Encoder provides a simple, high-performance API for quantum data encoding. The library is designed to be:

- **Fast**: 40-90x speedup over NumPy through SIMD optimization
- **Simple**: Minimal API for common use cases
- **Pythonic**: Follows Python conventions and type hints
- **Type-Safe**: Comprehensive type hints for IDE support

## Installation

```bash
# Build from source
git clone https://github.com/hybriq/simd_angle_encoder.git
cd simd_angle_encoder
./build.sh

# Or install with pip (future)
pip install simd-angle-encoder
```

## Quick Start

```python
import numpy as np
from simd_angle_encoder import encode, encode_batch, simd_info

# Check SIMD support
print(simd_info())

# Single encoding
data = np.array([0.1, 0.2, 0.3, 0.4])
encoded = encode(data, n_qubits=4)
print(encoded)
# Output: [0.62831853 1.25663706 1.88495559 2.51327412]

# Batch encoding
batch = np.random.random((100, 8))
encoded_batch = encode_batch(batch, n_qubits=10)
print(encoded_batch.shape)
# Output: (100, 10)

# Benchmark performance
from simd_angle_encoder import benchmark
numpy_time, simd_time = benchmark(data_size=128, batch_size=100, n_qubits=10)
speedup = numpy_time / simd_time
print(f"Speedup: {speedup:.2f}x")
```

## Core Functions

### encode()

Encode data using SIMD-optimized quantum angle encoding.

**Signature**:
```python
def encode(
    data: Union[np.ndarray, list, tuple],
    n_qubits: int
) -> np.ndarray:
    ...
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | Input data to encode. Will be converted to `np.ndarray` with `dtype=np.float64`. Can be 1D or will be flattened if multi-dimensional. |
| `n_qubits` | int | Number of qubits for encoding. Must be positive. If `len(data) < n_qubits`, result will be zero-padded. |

**Returns**:

- `np.ndarray`: Encoded angles in $[0, 2\pi]$ with shape `(n_qubits,)`

**Raises**:

- `TypeError`: If data cannot be converted to `np.ndarray`
- `ValueError`: If `n_qubits <= 0`

**Examples**:

Basic usage:
```python
import numpy as np
from simd_angle_encoder import encode

data = np.array([0.0, 0.25, 0.5, 0.75])
encoded = encode(data, n_qubits=4)

print(encoded)
# Output: [0.         1.57079633 3.14159265 4.71238898]
# Each value = data[i] * 2π
```

From list:
```python
# Automatically converts from list
data = [0.1, 0.2, 0.3]
encoded = encode(data, n_qubits=5)
print(encoded)
# Output: [0.62831853 1.25663706 1.88495559 0.         0.        ]
# Note: Zero-padded to n_qubits=5
```

Multi-dimensional input (automatically flattened):
```python
data = np.random.random((4, 4))  # 2D array
encoded = encode(data, n_qubits=16)
print(encoded.shape)
# Output: (16,)
```

**Performance**:

| Data Size | Batch Size | Speedup vs NumPy | Throughput |
|-----------|------------|------------------|------------|
| 8 elements | 1 | 2-4x | ~50K ops/s |
| 64 elements | 10 | 10-15x | ~260K ops/s |
| 128 elements | 100 | 30-35x | ~770K ops/s |
| 256 elements | 1000 | 40-90x | ~1.7M ops/s |

**Notes**:

- Angle encoding maps data to rotation angles: $\theta_i = x_i \times 2\pi$
- Each angle can be applied to a qubit using RY rotation: $R_y(\theta_i)|0\rangle$
- Values are clipped to $[0, 1]$ before encoding (not enforced, but typical)
- Result is in range $[0, 2\pi]$ for input in range $[0, 1]$

**See Also**:

- `encode_batch()`: Encode multiple data vectors
- `benchmark()`: Compare performance vs NumPy

---

### encode_batch()

Encode a batch of data using SIMD-optimized quantum angle encoding.

**Signature**:
```python
def encode_batch(
    batch_data: Union[np.ndarray, list, tuple],
    n_qubits: int
) -> np.ndarray:
    ...
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch_data` | array-like | Batch data with shape `(batch_size, data_dim)`. Will be converted to `np.ndarray` with `dtype=np.float64`. If 1D, will be reshaped to `(1, len(data))`. |
| `n_qubits` | int | Number of qubits for encoding. Must be positive. |

**Returns**:

- `np.ndarray`: Encoded batch with shape `(batch_size, n_qubits)`

**Raises**:

- `TypeError`: If data cannot be converted to `np.ndarray`
- `ValueError`: If `n_qubits <= 0` or data cannot be reshaped to 2D

**Examples**:

Basic usage:
```python
import numpy as np
from simd_angle_encoder import encode_batch

# Create batch of data
batch = np.random.random((100, 8))
encoded_batch = encode_batch(batch, n_qubits=10)

print(encoded_batch.shape)
# Output: (100, 10)

print(encoded_batch[0])  # First encoded vector
# Output: [5.234 2.145 1.234 ...]  # 10 angles
```

From list of lists:
```python
# Automatically converts and validates
batch = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
]
encoded = encode_batch(batch, n_qubits=4)

print(encoded.shape)
# Output: (2, 4)
```

1D array (automatically reshaped):
```python
# 1D array is treated as single sample
data = np.array([0.1, 0.2, 0.3, 0.4])
encoded = encode_batch(data, n_qubits=4)

print(encoded.shape)
# Output: (1, 4)
```

**Performance**:

Batch encoding achieves significantly higher speedup than single encoding:

| Batch Size | Speedup vs NumPy | Amortized Overhead |
|------------|------------------|-------------------|
| 1 | 2-4x | High (44% overhead) |
| 10 | 10-15x | Medium (14% overhead) |
| 100 | 30-35x | Low (3% overhead) |
| 1000 | 40-90x | Very low (1% overhead) |

**Notes**:

- Python-Rust boundary overhead is amortized across batch
- Best performance for batch sizes >= 100
- Memory usage: `batch_size × n_qubits × 8` bytes (float64)

**See Also**:

- `encode()`: Encode single data vector
- `benchmark()`: Compare performance

---

### simd_info()

Get information about SIMD support and system configuration.

**Signature**:
```python
def simd_info() -> str:
    ...
```

**Returns**:

- `str`: Multi-line string with system information

**Examples**:

```python
from simd_angle_encoder import simd_info

print(simd_info())
# Output:
# SIMD Support: Yes
# System: darwin/arm64
# CPU Cores: 8
```

**Notes**:

- SIMD support is detected at compile time, not runtime
- "SIMD Support: Yes" means the library was compiled with SIMD flags
- Different architectures have different SIMD capabilities:
  - **x86-64**: SSE2, AVX, AVX2, AVX-512
  - **ARM64**: NEON

---

### benchmark()

Run benchmark comparing NumPy vs SIMD performance.

**Signature**:
```python
def benchmark(
    data_size: int,
    batch_size: int,
    n_qubits: int,
    n_runs: int = 10
) -> Tuple[float, float]:
    ...
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_size` | int | Size of data to encode (number of features) |
| `batch_size` | int | Batch size to use for batch encoding |
| `n_qubits` | int | Number of qubits |
| `n_runs` | int | Number of benchmark runs (default: 10) |

**Returns**:

- `Tuple[float, float]`: `(numpy_time_ms, simd_time_ms)` - Execution times in milliseconds

**Examples**:

```python
from simd_angle_encoder import benchmark

# Run benchmark
numpy_time, simd_time = benchmark(
    data_size=128,
    batch_size=100,
    n_qubits=10,
    n_runs=10
)

print(f"NumPy time: {numpy_time:.2f} ms")
print(f"SIMD time: {simd_time:.2f} ms")

speedup = numpy_time / simd_time
print(f"Speedup: {speedup:.2f}x")
# Output:
# NumPy time: 4.12 ms
# SIMD time: 0.13 ms
# Speedup: 32.45x
```

**Notes**:

- Benchmark includes warmup runs to avoid cold start effects
- Times are averages over `n_runs` runs
- Results vary by system (CPU, SIMD support, etc.)
- Use `n_runs >= 10` for stable results

---

## Type Definitions

### Input Types

The library accepts multiple input types for flexibility:

```python
from typing import Union, List, Tuple
import numpy as np

# All of these work:
data1: np.ndarray = np.array([0.1, 0.2, 0.3])
data2: List[float] = [0.1, 0.2, 0.3]
data3: Tuple[float, ...] = (0.1, 0.2, 0.3)

# All convert to np.ndarray internally
encoded = encode(data1, n_qubits=3)  # Works
encoded = encode(data2, n_qubits=3)  # Works
encoded = encode(data3, n_qubits=3)  # Works
```

### Output Types

All functions return `np.ndarray` with `dtype=np.float64`:

```python
encoded: np.ndarray = encode(data, n_qubits=4)
assert encoded.dtype == np.float64
```

**Future**: Complex arrays for amplitude encoding:
```python
# Phase 2: Amplitude encoding returns complex
# encoded: np.ndarray = amplitude_encode(data, n_qubits=2)
# assert encoded.dtype == np.complex128
```

---

## Error Handling

### Type Errors

```python
from simd_angle_encoder import encode

# Invalid type
try:
    encode("not an array", n_qubits=4)
except TypeError as e:
    print(e)
    # Output: data must be array-like (np.ndarray, list, tuple), got str
```

### Value Errors

```python
# Invalid n_qubits
try:
    encode(data, n_qubits=0)
except ValueError as e:
    print(e)
    # Output: n_qubits must be positive, got 0
```

### Dimension Errors

```python
import numpy as np

# encode() expects 1D or will flatten
data = np.random.random((4, 4))
encoded = encode(data, n_qubits=16)  # OK - flattened to 1D

# encode_batch() expects 2D or will reshape
batch = np.random.random(100)
encoded = encode_batch(batch, n_qubits=10)  # OK - reshaped to (1, 100)
```

---

## Performance Guidelines

### Best Practices

1. **Use batch encoding when possible**:
   ```python
   # Good: Batch encoding (30-90x speedup)
   batch = np.random.random((1000, 128))
   encoded = encode_batch(batch, n_qubits=128)

   # Avoid: Loop of single encodings (2-4x speedup)
   encoded = [encode(sample, n_qubits=128) for sample in batch]
   ```

2. **Pre-allocate arrays**:
   ```python
   # Good: Pre-allocate
   batch = np.empty((1000, 128), dtype=np.float64)
   # ... fill batch ...
   encoded = encode_batch(batch, n_qubits=128)

   # Avoid: Appending to list (slow)
   batch = []
   for _ in range(1000):
       batch.append(np.random.random(128))
   encoded = encode_batch(np.array(batch), n_qubits=128)
   ```

3. **Use correct dtype**:
   ```python
   # Good: Already float64
   data = np.array([0.1, 0.2, 0.3], dtype=np.float64)

   # Avoid: Type conversion (small overhead)
   data = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # Wrong dtype
   encoded = encode(data.astype(np.float64), n_qubits=3)  # Converts
   ```

### Performance Optimization

**Batch Size Selection**:

| Use Case | Recommended Batch Size | Speedup |
|----------|------------------------|---------|
| Real-time | 1-10 | 2-15x |
| Interactive | 10-100 | 15-35x |
| Batch Processing | 100-1000 | 35-90x |

**Memory vs Speed Trade-off**:

```python
# Larger batches = faster but more memory
batch = np.random.random((10000, 256))  # 25 MB
encoded = encode_batch(batch, n_qubits=256)  # Very fast (~80x)

# Smaller batches = slower but less memory
batch = np.random.random((100, 256))  # 0.25 MB
encoded = encode_batch(batch, n_qubits=256)  # Fast (~35x)
```

---

## Advanced Usage

### Integration with Quantum Frameworks

**PennyLane** (Phase 3):
```python
import pennylane as qml
from pennylane_simd_angle import AngleEncoding

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def circuit(data):
    AngleEncoding(data, wires=range(4))
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

data = np.array([0.1, 0.2, 0.3, 0.4])
result = circuit(data)
```

**Qiskit** (Phase 3):
```python
from qiskit import QuantumCircuit
from qiskit_simd_angle import angle_encode_circuit

data = np.array([0.1, 0.2, 0.3, 0.4])
qc = angle_encode_circuit(data, n_qubits=4)
qc.draw()
```

### Custom Encoding Functions

```python
from simd_angle_encoder._simd_angle_encoder import angle_encode_simd

# Direct Rust access (framework integration)
data = np.array([0.1, 0.2, 0.3, 0.4])
encoded = angle_encode_simd(data, n_qubits=4)
```

---

## Changelog

### Version 0.1.0 (Current)

**Added**:
- `encode()`: Single vector angle encoding
- `encode_batch()`: Batch angle encoding
- `simd_info()`: System information
- `benchmark()`: Performance benchmarking

**Performance**:
- 40-90x speedup vs NumPy for batch encoding
- 2-4x speedup for single encoding

### Version 0.2.0 (Planned - Phase 2)

**Planned**:
- `encode(method='amplitude')`: Amplitude encoding
- `encode(method='basis')`: Basis encoding
- 20-60x speedup for new encoding methods

### Version 0.3.0 (Planned - Phase 3)

**Planned**:
- PennyLane integration (`pennylane-simd-angle-encoder`)
- Qiskit integration (`qiskit-simd-angle-encoder`)

---

## Support

- **Documentation**: `/Users/syahriza/data/kubitto/simd-angle-encoder/docs/`
- **Issues**: https://github.com/hybriq/simd_angle_encoder/issues
- **Contributing**: See `CONTRIBUTING.md`

---

**End of Core API Documentation**
