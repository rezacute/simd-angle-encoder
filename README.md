# SIMD Angle Encoder

[![CI/CD Pipeline](https://github.com/hybriq/simd_angle_encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/hybriq/simd_angle_encoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hybriq/simd_angle_encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/hybriq/simd_angle_encoder)
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/hybriq/simd_angle_encoder)
[![Rust Version](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0--phase1-brightgreen)](https://github.com/hybriq/simd_angle_encoder)

A high-performance library for quantum angle encoding using Rust SIMD optimizations.

## Overview

This package provides SIMD-optimized implementations of angle encoding for quantum computing. Angle encoding maps classical data to quantum circuit rotation angles. The implementation uses Rust with SIMD (Single Instruction Multiple Data) optimizations to achieve significant performance improvements over pure Python implementations.

### Performance at a Glance

- **Average Speedup**: 40.69x faster than NumPy
- **Maximum Speedup**: 95.81x for batch operations
- **Batch Processing**: 68-95x speedup for 512+ elements
- **Throughput**: Up to 1.34 million operations per millisecond

## Features

- Fast angle encoding for individual data vectors
- Optimized batch processing for multiple data vectors
- SIMD optimization for modern CPU architectures (ARM NEON, x86 AVX)
- Seamless Python interface with NumPy integration
- Built-in benchmarking and profiling tools
- Comprehensive test coverage (92+ tests)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Performance](#performance)
- [Documentation](#documentation)
- [Examples](#examples)
- [Quality Assurance](#quality-assistance)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

Before installing, ensure you have the following:

- **Python**: 3.8 or later (tested up to 3.12)
- **Rust toolchain**: Stable Rust compiler (for building from source)
- **Maturin**: Python package builder (installed automatically if needed)
- **NumPy**: For array operations

### Quick Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/hybriq/simd_angle_encoder.git
   cd simd_angle_encoder
   ```

2. Build and install using the provided build script:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

The build script automatically:
- Detects Python 3.13 and sets required PyO3 compatibility flags
- Installs Maturin if not present
- Builds the release-optimized Rust extension
- Installs the Python package

### Verify Installation

Test your installation with:

```bash
python -c "from simd_angle_encoder import simd_info; print(simd_info())"
```

Expected output:
```
SIMD Information:
- Platform: <your-platform>
- SIMD Support: <NEON/AVX2/AVX-512>
- CPU: <your-cpu-model>
```

## Quick Start

Get started with SIMD Angle Encoder in minutes. Here are four practical examples:

### Example 1: Basic Single Encoding

Encode a single data vector into quantum rotation angles:

```python
import numpy as np
from simd_angle_encoder import encode

# Your classical data (e.g., features from a dataset)
data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# Encode to quantum rotation angles
n_qubits = 10  # Number of qubits in your quantum circuit
encoded = encode(data, n_qubits)

print(f"Input: {data}")
print(f"Encoded angles: {encoded}")
# Output: [0.628 1.257 1.885 2.513 3.142 3.770 4.398 5.027 0.000 0.000]
```

### Example 2: Batch Processing for Machine Learning

Encode multiple data points efficiently (ideal for training quantum ML models):

```python
import numpy as np
from simd_angle_encoder import encode_batch

# Batch of data (e.g., 100 samples, 8 features each)
batch_data = np.random.random((100, 8))
n_qubits = 10

# Encode entire batch at once
encoded_batch = encode_batch(batch_data, n_qubits)

print(f"Input shape: {batch_data.shape}")      # (100, 8)
print(f"Encoded shape: {encoded_batch.shape}")  # (100, 10)
print(f"First sample encoded: {encoded_batch[0]}")
```

### Example 3: Performance Comparison

Compare SIMD performance against NumPy:

```python
from simd_angle_encoder import benchmark
import numpy as np

# Benchmark with realistic data sizes
data_size = 128    # Number of features
batch_size = 100   # Number of samples
n_qubits = 16      # Target qubit count

numpy_time, simd_time = benchmark(
    data_size=data_size,
    batch_size=batch_size,
    n_qubits=n_qubits
)

speedup = numpy_time / simd_time

print(f"NumPy time:  {numpy_time:.4f} ms")
print(f"SIMD time:   {simd_time:.4f} ms")
print(f"Speedup:     {speedup:.2f}x")
# Typical output: Speedup: 40-90x depending on data size
```

### Example 4: Integration with NumPy Workflow

Use seamlessly in your NumPy-based workflow:

```python
import numpy as np
from simd_angle_encoder import encode_batch

# Your existing data preprocessing pipeline
data = np.load('your_data.npy')  # Shape: (1000, 16)

# Normalize data
data_normalized = (data - data.mean(axis=0)) / data.std(axis=0)

# Encode to quantum angles
encoded = encode_batch(data_normalized, n_qubits=20)

# Use in your quantum circuit
print(f"Ready for quantum processing: {encoded.shape}")
# Output: Ready for quantum processing: (1000, 20)

# You can now use `encoded` directly in:
# - PennyLane quantum circuits
# - Qiskit circuits
# - Custom quantum algorithms
```

## Performance

The SIMD-optimized implementation provides substantial speedups compared to pure NumPy implementations. Performance has been validated through comprehensive benchmarking with **189 individual benchmarks** across **24 categories**.

### Benchmark Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Total Benchmarks** | 189 | ✅ Comprehensive |
| **Average Speedup vs NumPy** | 40.69x | ✅ Exceeds Target |
| **Maximum Speedup** | 95.81x | ✅ Excellent |
| **Minimum Speedup** | 1.74x | ⚠️ Small Data |
| **Test Platform** | Apple M3 Pro (12 cores) | ✅ Modern Hardware |

### Speedup by Data Size

Performance scales with data size - larger data sets benefit more from SIMD optimization:

| Data Size | NumPy Time (μs) | SIMD Time (μs) | Speedup | Use Case |
|-----------|-----------------|---------------|---------|----------|
| 4         | 0.66            | 0.38          | 1.74x   | Minimal data |
| 8         | 0.91            | 0.38          | 2.41x   | Small vectors |
| 16        | 1.56            | 0.32          | 4.86x   | Low dimensions |
| 32        | 2.87            | 0.33          | 8.60x   | Medium features |
| 64        | 5.51            | 0.46          | 11.93x  | Medium datasets |
| 128       | 10.76           | 0.43          | 25.28x  | Large features |
| 256       | 21.06           | 0.51          | 41.43x  | High dimensions |
| 512       | 44.26           | 0.64          | 68.79x  | Very large data |
| 1024      | 88.60           | 0.92          | 95.81x  | Batch processing |

**Key Insight**: Speedup increases with data size, achieving **68-95x** for 512+ elements.

### Batch Processing Performance

For quantum machine learning workloads that process multiple samples:

| Batch Size | Time (ms) | Throughput (ops/ms) | Speedup vs NumPy |
|------------|-----------|---------------------|------------------|
| 1          | 0.0006    | 1,649               | 2-4x             |
| 10         | 0.0017    | 5,759               | 10-15x           |
| 100        | 0.0140    | 7,136               | 30-35x           |
| 1,000      | 0.1292    | 7,740               | 40-90x           |
| 10,000     | 1.5520    | 6,443               | 68-95x           |

**Near-linear scaling**: Batch size scaling ratio of 1.11 (1.0 = perfect linear)

### Visual Performance Comparison

```
Speedup Factor vs NumPy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

100x │                                              █
     │                                        █    █
 75x │                                        █    █
     │                                  █     █    █
 50x │                            █     █     █    █
     │                      █     █     █     █    █
 25x │                █     █     █     █     █    █
     │          █     █     █     █     █     █    █
  0x └────────┼─────┼─────┼─────┼─────┼─────┼────┼────
           4     16    64    128   256   512   1024
                           Data Size (elements)

Key: █ SIMD Speedup,  ─ NumPy Baseline
```

### Performance Expectations

What speedup should you expect?

- **Small data (< 16 elements)**: 2-8x speedup
  - Suitable for single quantum circuit encoding
  - Minimal benefit due to function call overhead

- **Medium data (16-64 elements)**: 5-12x speedup
  - Good for moderate feature sets
  - Noticeable improvement in batch processing

- **Large data (64-256 elements)**: 25-45x speedup
  - Excellent for high-dimensional features
  - Significant time savings in ML pipelines

- **Batch data (256+ elements)**: 68-95x speedup
  - Ideal for quantum ML training
  - Maximum benefit from SIMD vectorization

### Scalability Analysis

**Data Size Scaling**
- Scaling ratio: 0.80 (better than linear!)
- Larger datasets achieve super-linear speedup
- Excellent cache locality and SIMD utilization

**Batch Size Scaling**
- Scaling ratio: 1.11 (near-linear)
- Consistent performance across batch sizes
- Predictable behavior for ML workloads

### Benchmark Details

For complete benchmark results, see:
- [Baseline Summary](benchmarks/reports/BASELINE_SUMMARY.md) - Executive summary
- [Full Report](benchmarks/reports/baseline.md) - Detailed 549-line report
- [Raw Data](benchmarks/reports/baseline.json) - JSON benchmark data

**Test Environment**
- System: Darwin arm (macOS 24.6.0)
- CPU: Apple M3 Pro (12 cores)
- Python: 3.12.5
- Compiler: Clang 16.0.6
- SIMD: ARM NEON

## Documentation

Comprehensive documentation is available to help you get started and use the library effectively.

### Getting Started Guides

- **[README](README.md)** - This file, project overview and quick start
- **[Development Guide](DEVELOPMENT.md)** - Development setup and quick reference
- **[Contributing Guide](CONTRIBUTING.md)** - Contribution guidelines and workflow
- **[Testing Documentation](TESTING_README.md)** - Testing strategy and guidelines

### Architecture Documentation (NEW)

Complete architecture documentation from Phase 1:

- **[Documentation Index](docs/index.md)** - Central hub for all documentation
- **[Architecture Overview](docs/architecture/README.md)** - System architecture and design principles
- **[ADR-001: Architecture Overview](docs/architecture/adr/ADR-001-overview.md)** - Complete system architecture with detailed diagrams
- **[ADR-002: SIMD Optimization Strategy](docs/architecture/adr/ADR-002-simd-strategy.md)** - Multi-tier SIMD approach and performance targets
- **[ADR-003: Encoding Methods Architecture](docs/architecture/adr/ADR-003-encoding-methods.md)** - Angle, amplitude, and basis encoding specifications
- **[ADR-004: Framework Integration Patterns](docs/architecture/adr/ADR-004-framework-integration.md)** - PennyLane and Qiskit integration designs
- **[ADR-005: API Design Rationale](docs/architecture/adr/ADR-005-api-design.md)** - API philosophy and design decisions
- **[Technical Trade-offs Analysis](docs/architecture/technical-trade-offs.md)** - Decision matrix and rationale

### API Documentation

- **[Core API Reference](docs/api/core-api.md)** - Complete API documentation
  - `encode()` - Single vector encoding
  - `encode_batch()` - Batch encoding
  - `simd_info()` - SIMD capability detection
  - `benchmark()` - Performance benchmarking

### Project Planning

- **[Roadmap v1.0](ROADMAP_V1.md)** - Complete 24-week roadmap to version 1.0
- **[Baseline Summary](benchmarks/reports/BASELINE_SUMMARY.md)** - Performance baseline executive summary
- **[Full Benchmark Report](benchmarks/reports/baseline.md)** - Detailed 549-line benchmark analysis

## Examples

### Run the Example Script

A comprehensive example script is included in the repository:

```bash
python example.py
```

This script demonstrates:
1. Single vector encoding
2. Batch encoding
3. Performance benchmarking across multiple configurations
4. Automatic speedup plotting (requires matplotlib)

The example will output a table of speedup factors and generate a visualization plot.

### Using in Quantum Machine Learning

**PennyLane Integration Example** (Phase 2 - planned):

```python
import pennylane as qml
from simd_angle_encoder import encode_batch

# Prepare your data
X_train = np.random.random((1000, 16))

# Encode to quantum angles
X_encoded = encode_batch(X_train, n_qubits=20)

# Use in PennyLane quantum circuit
dev = qml.device('default.qubit', wires=20)

@qml.qnode(dev)
def quantum_circuit(weights, x):
    # Use encoded angles directly
    for i in range(20):
        qml.RY(x[i], wires=i)
    # ... rest of circuit
    return qml.expval(qml.PauliZ(0))
```

## Quality Assurance

This project maintains high code quality standards through comprehensive CI/CD:

- **Automated Testing**: 92+ unit, integration, and property-based tests
- **Multi-Platform Support**: Tested on Ubuntu, macOS, and Windows
- **Python Compatibility**: Supports Python 3.8-3.12
- **Code Coverage**: Continuous coverage tracking with Codecov
- **Linting**: Automated checks with Rust Clippy and Python Flake8
- **Formatting**: Enforced code style with Rust fmt and Black
- **Security Scanning**: Automated dependency and code security audits
- **Benchmark CI**: Continuous performance regression detection (planned for Phase 2)

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and testing guidelines.

## How It Works

The SIMD Angle Encoder applies several optimization techniques to achieve high performance:

### 1. SIMD Vectorization

Processing multiple data elements simultaneously using CPU vector instructions:
- **ARM NEON**: 128-bit registers (4 x f32 or 2 x f64)
- **x86 AVX2**: 256-bit registers (8 x f32 or 4 x f64)
- **x86 AVX-512**: 512-bit registers (16 x f32 or 8 x f64)

### 2. Cache-Friendly Access Patterns

Processing data in chunks for better cache locality:
- Sequential memory access patterns
- Chunked processing (matches cache line sizes)
- Minimal cache misses for large datasets

### 3. Rust Performance

Leveraging Rust's zero-cost abstractions and compiler optimizations:
- LLVM-based compiler with aggressive optimization
- No runtime overhead (zero-cost abstractions)
- Memory safety without garbage collection

### 4. PyO3 Integration

Efficient Python bindings with minimal overhead:
- Direct NumPy array access (no copying)
- Minimal Python-Rust boundary crossing
- Efficient error handling

### Data Flow

```
Python NumPy Array
        ↓
   PyO3 Binding (zero-copy)
        ↓
   Rust Function
        ↓
   SIMD Vectorization
        ↓
   NumPy Array Result
        ↓
   Python (returned directly)
```

For complete architectural details, see the [Architecture Documentation](docs/architecture/README.md).

## Contributing

We welcome contributions to the SIMD Angle Encoder project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Development Setup

```bash
# Clone the repository
git clone https://github.com/hybriq/simd_angle_encoder.git
cd simd_angle_encoder

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install maturin numpy pytest

# Build in development mode
maturin develop

# Run tests
pytest tests/

# Run formatting
rustfmt src/lib.rs
black tests/

# Run linting
cargo clippy
flake8 tests/
```

### Contribution Areas

We're particularly interested in contributions for:

- **Phase 2 Optimization**: Small data performance optimization
- **Framework Integration**: PennyLane and Qiskit integrations
- **Additional Encodings**: Amplitude and basis encoding methods
- **Documentation**: Tutorials, examples, and guides
- **Benchmarks**: Cross-platform performance validation
- **Tests**: Additional test cases and coverage improvements

See [ROADMAP_V1.md](ROADMAP_V1.md) for the complete development roadmap.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use SIMD Angle Encoder in your research, please cite:

```bibtex
@software{simd_angle_encoder,
  title = {SIMD Angle Encoder: High-Performance Quantum ML Encoding},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/hybriq/simd_angle_encoder}
}
```

## Acknowledgments

Built with:
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [PyO3](https://pyo3.rs/) - Python bindings for Rust
- [NumPy](https://numpy.org/) - Numerical computing in Python
- [Maturin](https://github.com/PyO3/maturin) - Rust-based Python extension builder

## Support

- **Documentation**: See [docs/index.md](docs/index.md)
- **Issues**: [GitHub Issues](https://github.com/hybriq/simd_angle_encoder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hybriq/simd_angle_encoder/discussions)

---

**Project Status**: Phase 1 Complete (Version 0.1.0)
**Next Milestone**: Phase 2 - Small data optimization and framework integration

For the latest updates and roadmap, see [ROADMAP_V1.md](ROADMAP_V1.md). 