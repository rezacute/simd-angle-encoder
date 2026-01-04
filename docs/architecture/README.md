# SIMD Angle Encoder - Architecture Documentation

This directory contains comprehensive architecture documentation and Architectural Decision Records (ADRs) for the SIMD Angle Encoder project.

## Overview

The SIMD Angle Encoder is a high-performance quantum machine learning library that provides SIMD-optimized implementations of quantum data encoding methods. The architecture is designed around three core principles:

1. **Performance First**: Every design decision prioritizes maintaining or enhancing SIMD performance characteristics
2. **Zero-Copy Boundaries**: Minimize data copying between Python and Rust through PyO3's efficient integration
3. **Extensibility**: Modular architecture enabling easy addition of new encoding methods and framework integrations

## Architecture Documentation Structure

```
docs/architecture/
├── README.md                    # This file - architecture overview
├── adr/
│   ├── ADR-001-overview.md      # System architecture and design principles
│   ├── ADR-002-simd-strategy.md # SIMD optimization strategy and results
│   ├── ADR-003-encoding-methods.md # Encoding method architecture
│   ├── ADR-004-framework-integration.md # Framework integration patterns
│   └── ADR-005-api-design.md    # API design rationale and evolution
├── data-flow.md                 # Data flow diagrams and Python-Rust interaction
├── performance-model.md         # Performance characteristics and optimization
├── testing-strategy.md          # Testing architecture and quality assurance
└── module-structure.md          # Module organization and dependencies
```

## Quick Links

- **System Overview**: See [ADR-001: Architecture Overview](adr/ADR-001-overview.md)
- **SIMD Strategy**: See [ADR-002: SIMD Optimization Strategy](adr/ADR-002-simd-strategy.md)
- **Encoding Methods**: See [ADR-003: Encoding Methods Architecture](adr/ADR-003-encoding-methods.md)
- **Framework Integration**: See [ADR-004: Framework Integration Patterns](adr/ADR-004-framework-integration.md)
- **API Design**: See [ADR-005: API Design Rationale](adr/ADR-005-api-design.md)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                               │
│  (Python Applications: QML Workloads, Neural Networks, etc.)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FRAMEWORK LAYER                             │
│  (PennyLane Plugin, Qiskit Integration, Future: Cirq, etc.)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER                                  │
│  (Python Wrapper: encode(), encode_batch(), simd_info())        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PyO3 BOUNDARY                                │
│  (Efficient Python-Rust FFI with minimal overhead)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RUST CORE LAYER                                │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐ │
│  │ Angle Encode │ Amplitude    │ Basis Encode │ Future:     │ │
│  │ (SIMD)       │ Encode (SIMD)│ (SIMD)       │ Dense, etc. │ │
│  └──────────────┴──────────────┴──────────────┴─────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      SIMD Vectorization Engine (AVX2, AVX-512, NEON)     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CPU HARDWARE LAYER                            │
│  (x86-64 with AVX2/AVX-512, ARM64 with NEON)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Architectural Components

### 1. Core Encoding Engine (Rust)

**Location**: `/Users/syahriza/data/kubitto/simd-angle-encoder/src/lib.rs`

**Responsibilities**:
- SIMD-optimized encoding implementations
- Memory-efficient data processing
- Cache-friendly access patterns
- Multi-platform SIMD support (AVX2, AVX-512, NEON)

**Design Principles**:
- Vectorized operations using 4-element chunks (aligns with SIMD width)
- Zero-copy wherever possible
- Minimal allocations in hot paths

### 2. Python API Layer

**Location**: `/Users/syahriza/data/kubitto/simd-angle-encoder/python/simd_angle_encoder/__init__.py`

**Responsibilities**:
- User-friendly Python interface
- Input validation and type conversion
- NumPy integration
- Error handling and messaging

**Design Principles**:
- Minimal overhead wrapper
- Clear, intuitive API
- Comprehensive docstrings
- Type hints for IDE support

### 3. Framework Integration Layer

**Locations** (planned):
- PennyLane: `/pennylane-simd-angle-encoder/`
- Qiskit: `/qiskit-simd-angle-encoder/`

**Responsibilities**:
- Native framework operations
- Gradient computation support
- Framework-specific optimizations
- Documentation and examples

### 4. Testing & Benchmarking Infrastructure

**Locations**:
- Tests: `/Users/syahriza/data/kubitto/simd-angle-encoder/tests/`
- Benchmarks: `/Users/syahriza/data/kubitto/simd-angle-encoder/benchmarks/`

**Coverage**:
- Unit tests (individual functions)
- Integration tests (Python-Rust boundary)
- Property-based tests (invariants)
- Performance benchmarks (pytest-benchmark, Criterion)

## Performance Characteristics

### Current Performance (Angle Encoding)

| Batch Size | Speedup vs NumPy | Throughput |
|------------|------------------|------------|
| 1          | 2-4x             | ~1M ops/s  |
| 10         | 10-15x           | ~8M ops/s  |
| 100        | 30-35x           | ~25M ops/s |
| 1000       | 40-90x           | ~40M ops/s |

### Performance Optimization Strategy

1. **SIMD Vectorization**: Process 4-8 elements simultaneously using AVX2/AVX-512
2. **Cache Locality**: Process data in chunks that fit in L1/L2 cache
3. **Parallelization**: Use Rayon for multi-core parallel processing (planned)
4. **Zero-Copy**: Minimize memory allocations and copies
5. **Compiler Optimization**: Use Rust's release mode with aggressive optimizations

## Technology Stack

### Core Technologies

- **Rust**: Systems programming language for performance-critical code
- **PyO3**: Python-Rust bindings with minimal overhead
- **NumPy**: Array operations and Python integration
- **SIMD Intrinsics**: AVX2, AVX-512 (x86-64), NEON (ARM64)

### Development Tools

- **Testing**: pytest, pytest-cov, hypothesis
- **Benchmarking**: pytest-benchmark, Criterion (Rust)
- **CI/CD**: GitHub Actions
- **Code Quality**: rustfmt, clippy, black, flake8, mypy

## Design Patterns

### 1. Zero-Copy Pattern

Minimize data copying across the Python-Rust boundary:

```python
# Python side - uses read-only views
def encode(data: np.ndarray, n_qubits: int) -> np.ndarray:
    # Validation and conversion happens here
    return _angle_encode_simd(data, n_qubits)  # Rust handles actual encoding
```

```rust
// Rust side - efficient array access
fn angle_encode_simd<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,  // Read-only view, no copy
    n_qubits: usize,
) -> Bound<'py, PyArray1<f64>> {
    let data_array = data.as_array();  // Zero-copy view
    let result = simd_angle_encode(&data_array.to_vec(), n_qubits);
    result.into_pyarray(py)  // Move into Python, no copy
}
```

### 2. SIMD Chunking Pattern

Process data in fixed-size chunks for optimal vectorization:

```rust
const CHUNK_SIZE: usize = 4;  // Aligns with SIMD width

while i + CHUNK_SIZE <= data.len() && i < n_qubits {
    for j in 0..CHUNK_SIZE {
        if i + j < n_qubits {
            result.push(data[i + j] * two_pi);
        }
    }
    i += CHUNK_SIZE;
}
```

### 3. Layered Architecture

Separate concerns across distinct layers:

- **User Layer**: Framework-specific code
- **API Layer**: Python interface
- **Core Layer**: Rust implementations
- **Hardware Layer**: CPU-specific optimizations

### 4. Extension Pattern

Easy addition of new encoding methods:

```rust
// New encoding method follows same pattern
pub fn new_encoding_method_simd(data: &[f64], params: EncodingParams) -> Vec<f64> {
    // SIMD-optimized implementation
}

// Python wrapper follows same pattern
#[pyfunction]
fn new_encoding_method<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    params: EncodingParams,
) -> Bound<'py, PyArray1<f64>> {
    // Validation, delegation to Rust, return result
}
```

## Module Dependencies

```
┌────────────────────────────────────────────────────────────┐
│                    Python Modules                          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ simd_angle_encoder (Core API)                        │ │
│  │  - encode()                                          │ │
│  │  - encode_batch()                                    │ │
│  │  - simd_info()                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                           ↓                               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Framework Integrations (Future)                      │ │
│  │  - pennylane_simd_angle                              │ │
│  │  - qiskit_simd_angle                                 │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│                    Rust Core                               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ lib.rs (Module Root)                                 │ │
│  │  - PyO3 module definition                            │ │
│  │  - Function exports                                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                           ↓                               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Encoding Implementations (Future modules)            │ │
│  │  - angle.rs (Angle encoding)                         │ │
│  │  - amplitude.rs (Amplitude encoding)                 │ │
│  │  - basis.rs (Basis encoding)                         │ │
│  └──────────────────────────────────────────────────────┘ │
│                           ↓                               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ SIMD Engine (Future)                                │ │
│  │  - simd_ops.rs (Vectorized operations)               │ │
│  │  - arch/ (Architecture-specific code)                │ │
│  │    - x86_64.rs (AVX2, AVX-512)                       │ │
│  │    - arm64.rs (NEON)                                 │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

## Future Architectural Evolution

### Phase 2: Core Encoding (Weeks 5-10)

**Planned Changes**:
- Refactor into module-based structure
- Add amplitude and basis encoding modules
- Implement unified SIMD engine
- Add architecture-specific optimizations

**New Structure**:
```
src/
├── lib.rs              # Module root and PyO3 bindings
├── encoders/
│   ├── mod.rs
│   ├── angle.rs        # Angle encoding (extracted from lib.rs)
│   ├── amplitude.rs    # Amplitude encoding (new)
│   └── basis.rs        # Basis encoding (new)
└── simd/
    ├── mod.rs
    ├── ops.rs          # Generic SIMD operations
    └── arch/
        ├── x86_64.rs   # AVX2, AVX-512 implementations
        └── arm64.rs    # NEON implementations
```

### Phase 3: Framework Integration (Weeks 11-15)

**Planned Changes**:
- Separate packages for each framework
- Plugin architecture for PennyLane
- Circuit builders for Qiskit
- Shared integration utilities

**New Packages**:
```
pennylane-simd-angle-encoder/
├── pennylane_simd_angle/
│   ├── __init__.py
│   ├── angle.py
│   ├── amplitude.py
│   └── basis.py
└── tests/

qiskit-simd-angle-encoder/
├── qiskit_simd_angle/
│   ├── __init__.py
│   ├── encoders/
│   └── circuits/
└── tests/
```

## Architectural Principles

### 1. SIMD-First Design

Every encoding method must be designed with SIMD optimization in mind from the start.

**Implications**:
- Prefer fixed-size data structures
- Use contiguous memory layouts
- Avoid branching in hot paths
- Design algorithms that process data in parallel

### 2. Zero-Copy Boundaries

Minimize data copying across the Python-Rust FFI boundary.

**Implications**:
- Use PyO3's read-only array views
- Prefer references over owned data
- Allocate results directly in Python's memory
- Avoid serialization/deserialization

### 3. Cache Locality

Structure data to maximize cache hits.

**Implications**:
- Process data in chunks that fit in L1 cache (typically 32-64 KB)
- Use sequential access patterns
- Prefer structure-of-arrays over array-of-structures when beneficial
- Align data to cache line boundaries (64 bytes)

### 4. Parallelizable by Default

Design algorithms that can be parallelized.

**Implications**:
- Avoid shared mutable state
- Use functional programming patterns
- Design for data parallelism (map-reduce style)
- Consider Rayon for Rust parallelization

### 5. Testability First

Design for testability from the ground up.

**Implications**:
- Pure functions wherever possible
- Dependency injection for external dependencies
- Clear separation of concerns
- Comprehensive test coverage

## Non-Goals

The following are explicitly **not** goals of this architecture:

1. **GPU Support**: Focused on CPU SIMD optimization (Phase 1-5)
2. **Distributed Computing**: Single-machine optimization only
3. **Real-time Guarantees**: Best-effort performance, no hard real-time constraints
4. **Generic Quantum Computing**: Focused on data encoding, not general quantum algorithms
5. **Alternative Language Bindings**: Python-only (C++, Julia, etc. not in scope)

## Maintenance Guidelines

### Adding New Encoding Methods

1. Create module in `src/encoders/`
2. Implement SIMD-optimized algorithm
3. Add Python wrapper in `lib.rs`
4. Add public API in `python/simd_angle_encoder/__init__.py`
5. Write comprehensive tests
6. Add benchmarks
7. Update ADR-003 (Encoding Methods)
8. Update this README

### Performance Regression Prevention

1. All benchmarks must run in CI
2. Performance must not degrade >5% without explicit approval
3. New features must include performance targets
4. Use Criterion for Rust benchmark comparisons
5. Track performance over time

### Documentation Updates

1. Every architectural change requires ADR update
2. API changes require API documentation updates
3. Performance changes require performance model updates
4. New modules require architecture diagram updates

## Related Documentation

- **Project Roadmap**: `/Users/syahriza/data/kubitto/simd-angle-encoder/ROADMAP_V1.md`
- **Contributing Guide**: `/Users/syahriza/data/kubitto/simd-angle-encoder/CONTRIBUTING.md`
- **Development Guide**: `/Users/syahriza/data/kubitto/simd-angle-encoder/DEVELOPMENT.md`
- **API Documentation**: `/Users/syahriza/data/kubitto/simd-angle-encoder/docs/api/`
- **Framework Integration**: `/Users/syahriza/data/kubitto/simd-angle-encoder/docs/integrations/`

## Document Metadata

**Version**: 1.0.0
**Last Updated**: 2026-01-04
**Author**: Architecture Agent (P1-TASK-007)
**Status**: Active
**Next Review**: End of Phase 2 (Week 10)
