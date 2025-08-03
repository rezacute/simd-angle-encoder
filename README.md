# SIMD Angle Encoder

A high-performance library for quantum angle encoding using Rust SIMD optimizations.

## Overview

This package provides SIMD-optimized implementations of angle encoding for quantum computing. Angle encoding maps classical data to quantum circuit rotation angles. The implementation uses Rust with SIMD (Single Instruction Multiple Data) optimizations to achieve significant performance improvements over pure Python implementations.

## Features

- Fast angle encoding for individual data vectors
- Optimized batch processing for multiple data vectors
- SIMD optimization for modern CPU architectures
- Seamless Python interface with NumPy integration
- Built-in benchmarking functionality

## Installation

### Prerequisites

- Python 3.8 or later
- Rust toolchain (for building from source)
- NumPy

### Building from source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simd_angle_encoder.git
   cd simd_angle_encoder
   ```

2. Build and install:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

For Python 3.13 users: The build script automatically sets the required PyO3 compatibility flag.

## Usage

```python
import numpy as np
from simd_angle_encoder import encode, encode_batch, simd_info

# Get SIMD support information
print(simd_info())

# Single vector encoding
data = np.random.random(8)
n_qubits = 10
encoded = encode(data, n_qubits)
print(encoded)

# Batch encoding for multiple vectors
batch_data = np.random.random((100, 8))
encoded_batch = encode_batch(batch_data, n_qubits)
print(encoded_batch.shape)  # (100, 10)

# Benchmarking against pure Python implementation
from simd_angle_encoder import benchmark
numpy_time, simd_time = benchmark(data_size=128, batch_size=100, n_qubits=10)
speedup = numpy_time / simd_time
print(f"Speedup: {speedup:.2f}x")
```

## Performance

The SIMD-optimized implementation provides substantial speedups compared to pure Python:

| Batch Size | Typical Speedup |
|------------|----------------|
| 1          | 2-4x           |
| 10         | 10-15x         |
| 100        | 30-35x         |
| 1000       | 40-90x         |

The performance advantage increases with batch size, making this implementation particularly valuable for quantum machine learning applications that process large batches of data.

## How It Works

The implementation applies several optimization techniques:

1. **SIMD Vectorization**: Processing multiple data elements simultaneously using CPU vector instructions
2. **Cache-Friendly Access Patterns**: Processing data in chunks for better cache locality
3. **Rust Performance**: Leveraging Rust's zero-cost abstractions and compiler optimizations
4. **PyO3 Integration**: Efficient Python bindings with minimal overhead

## License

MIT License 