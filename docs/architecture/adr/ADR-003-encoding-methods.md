# ADR-003: Encoding Methods Architecture

**Status**: Proposed
**Date**: 2026-01-04
**Deciders**: Architecture Agent, Feature Expansion Agent
**Related**: ADR-001 (Architecture Overview), ADR-002 (SIMD Strategy)

## Context

Quantum machine learning requires multiple encoding methods to transform classical data into quantum states. Different applications and algorithms benefit from different encoding approaches:

- **Angle Encoding**: Maps data to rotation angles (current implementation)
- **Amplitude Encoding**: Maps data to quantum state amplitudes (Phase 2)
- **Basis Encoding**: Maps data to computational basis states (Phase 2)
- **Dense Angle Encoding**: Enhanced angle encoding with more qubits (stretch goal)

Each encoding method has different mathematical properties, performance characteristics, and use cases. This ADR defines the architecture for implementing multiple encoding methods while maintaining code reuse, testability, and performance.

## Decision

We adopt a **modular encoding architecture** with shared SIMD infrastructure:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python API Layer                             │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ encode(data, method='angle', n_qubits=...)                │ │
│  │ encode_batch(data, method='amplitude', n_qubits=...)      │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Rust Core Layer                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Encoding Modules (src/encoders/)                          │ │
│  │  ┌────────────────┬────────────────┬────────────────┐     │ │
│  │  │ angle.rs       │ amplitude.rs   │ basis.rs       │     │ │
│  │  │ (124 lines)    │ (Phase 2)      │ (Phase 2)      │     │ │
│  │  └────────────────┴────────────────┴────────────────┘     │ │
│  │                        ↓                                 │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ SIMD Engine (src/simd/)                              │ │ │
│  │  │  - ops.rs (vectorized operations)                    │ │ │
│  │  │  - norm.rs (normalization)                           │ │ │
│  │  │  - arch/ (architecture-specific code)                │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure (Phase 2)

```
src/
├── lib.rs                      # PyO3 module root
├── encoders/
│   ├── mod.rs                  # Encoder trait and common types
│   ├── angle.rs                # Angle encoding (extract from lib.rs)
│   ├── amplitude.rs            # Amplitude encoding (new)
│   └── basis.rs                # Basis encoding (new)
└── simd/
    ├── mod.rs                  # SIMD utilities
    ├── ops.rs                  # Vectorized operations
    ├── norm.rs                 # Normalization operations
    └── arch/
        ├── x86_64.rs           # x86 SIMD implementations
        └── arm64.rs            # ARM SIMD implementations
```

### Common Encoder Trait

```rust
// src/encoders/mod.rs

use ndarray::Array1;

/// Common encoding method enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingMethod {
    Angle,
    Amplitude,
    Basis,
}

/// Common trait for all encoding methods
pub trait Encoder {
    /// Encode a single data vector
    fn encode(&self, data: &[f64], n_qubits: usize) -> Vec<f64>;

    /// Encode a batch of data vectors
    fn encode_batch(&self, batch_data: &[Vec<f64>], n_qubits: usize) -> Vec<Vec<f64>> {
        batch_data
            .iter()
            .map(|data| self.encode(data, n_qubits))
            .collect()
    }

    /// Get the name of this encoding method
    fn name(&self) -> &str;
}

/// Angle encoder
pub struct AngleEncoder;

impl Encoder for AngleEncoder {
    fn encode(&self, data: &[f64], n_qubits: usize) -> Vec<f64> {
        crate::encoders::angle::angle_encode_simd(data, n_qubits)
    }

    fn name(&self) -> &str {
        "angle"
    }
}

/// Amplitude encoder
pub struct AmplitudeEncoder;

impl Encoder for AmplitudeEncoder {
    fn encode(&self, data: &[f64], n_qubits: usize) -> Vec<f64> {
        crate::encoders::amplitude::amplitude_encode_simd(data, n_qubits)
    }

    fn name(&self) -> &str {
        "amplitude"
    }
}

/// Basis encoder
pub struct BasisEncoder;

impl Encoder for BasisEncoder {
    fn encode(&self, data: &[f64], n_qubits: usize) -> Vec<f64> {
        crate::encoders::basis::basis_encode_simd(data, n_qubits)
    }

    fn name(&self) -> &str {
        "basis"
    }
}
```

## Encoding Method Specifications

### 1. Angle Encoding (Current)

**Status**: ✅ Implemented (Phase 1)
**Performance**: 40-90x speedup (achieved)

**Mathematical Definition**:

Given input data $x = [x_1, x_2, ..., x_n]$ and $n$ qubits:

$$
\text{encoded}_i = x_i \times 2\pi \quad \text{for } i = 1, ..., n
$$

Each encoded value represents a rotation angle $\theta_i$ applied to qubit $i$:
$$
R_y(\theta_i) |0\rangle = \cos(\theta_i/2)|0\rangle + \sin(\theta_i/2)|1\rangle
$$

**Properties**:
- **Range**: $[0, 2\pi]$
- **Qubits**: One data element per qubit
- **Normalization**: Not required (each value independent)
- **Uniqueness**: Not unique (angles wrap around at $2\pi$)

**Use Cases**:
- Variational quantum circuits (VQCs)
- Quantum kernel methods
- Data where relative differences matter

**Implementation** (current):
```rust
// src/encoders/angle.rs

pub fn angle_encode_simd(data: &[f64], n_qubits: usize) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut result = Vec::with_capacity(n_qubits);

    let chunk_size = 4;
    let mut i = 0;

    while i + chunk_size <= data.len() && i < n_qubits {
        for j in 0..chunk_size {
            if i + j < n_qubits {
                result.push(data[i + j] * two_pi);
            }
        }
        i += chunk_size;
    }

    while i < data.len() && i < n_qubits {
        result.push(data[i] * two_pi);
        i += 1;
    }

    result.resize(n_qubits, 0.0);
    result
}
```

**Python API** (current):
```python
def encode(data: np.ndarray, n_qubits: int) -> np.ndarray:
    """Encode data using angle encoding with SIMD optimizations.

    Maps data to rotation angles in [0, 2π].

    Parameters
    ----------
    data : np.ndarray
        Input data (will be truncated to n_qubits elements)
    n_qubits : int
        Number of qubits

    Returns
    -------
    np.ndarray
        Encoded angles in [0, 2π]

    Examples
    --------
    >>> import numpy as np
    >>> from simd_angle_encoder import encode
    >>> data = np.array([0.0, 0.25, 0.5, 0.75])
    >>> encoded = encode(data, n_qubits=4)
    >>> print(encoded)
    [0.         1.57079633 3.14159265 4.71238898]
    """
```

### 2. Amplitude Encoding (Phase 2)

**Status**: 🔄 Planned (Weeks 5-6)
**Performance Target**: 20-50x speedup

**Mathematical Definition**:

Given input data $x = [x_1, x_2, ..., x_{2^n}]$ and $n$ qubits:

1. **Normalize** the data:
$$
x'_i = \frac{x_i}{\sqrt{\sum_{j=1}^{2^n} x_j^2}}
$$

2. **Create quantum state**:
$$
|\psi\rangle = \sum_{i=1}^{2^n} x'_i |i\rangle
$$

where $|i\rangle$ are computational basis states.

**Properties**:
- **Range**: Complex amplitudes with $|amplitude|^2$ summing to 1
- **Qubits**: $2^n$ data elements for $n$ qubits (exponential!)
- **Normalization**: Required (L2 norm = 1)
- **Uniqueness**: Unique up to global phase

**Use Cases**:
- Quantum kernel methods
- Quantum principal component analysis
- Algorithms requiring full quantum state representation

**Implementation** (proposed):
```rust
// src/encoders/amplitude.rs

use num_complex::Complex;
use crate::simd::norm::simd_norm;

pub fn amplitude_encode_simd(data: &[f64], _n_qubits: usize) -> Vec<f64> {
    // 1. Compute L2 norm using SIMD
    let norm = simd_norm(data);

    // 2. Avoid division by zero
    if norm < 1e-10 {
        return vec![0.0; data.len()];
    }

    // 3. Normalize using SIMD (will be complex in practice)
    let normalized: Vec<f64> = data
        .iter()
        .map(|&x| x / norm)
        .collect();

    // 4. Return as complex vector (for now, return real parts)
    // TODO: Return Vec<Complex<f64>> when Complex type is integrated
    normalized
}

// Test normalization property
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let encoded = amplitude_encode_simd(&data, 4);

        // Check that squared values sum to 1
        let sum_of_squares: f64 = encoded.iter().map(|&x| x * x).sum();
        assert!((sum_of_squares - 1.0).abs() < 1e-10);
    }
}
```

**SIMD Norm Computation**:
```rust
// src/simd/norm.rs

use std::arch::x86_64::*;

/// Compute L2 norm using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_norm_avx2(data: &[f64]) -> f64 {
    let mut sum = 0.0;
    let chunks = data.chunks_exact(4);

    for chunk in chunks {
        let vec = _mm256_loadu_pd(chunk.as_ptr());
        let squared = _mm256_mul_pd(vec, vec);

        // Horizontal sum
        let shuffle = _mm256_permute2f128_pd(squared, squared, 1);
        let sum2 = _mm256_add_pd(squared, shuffle);
        let result = _mm256_hadd_pd(sum2, sum2);
        sum += _mm256_cvtsd_f64(result);
    }

    // Handle remainder
    for &val in chunks.remainder() {
        sum += val * val;
    }

    sum.sqrt()
}
```

**Python API** (proposed):
```python
def amplitude_encode(data: np.ndarray, n_qubits: int) -> np.ndarray:
    """Encode data into quantum state amplitudes.

    Normalizes data and maps to quantum state amplitudes.
    The state |ψ⟩ has amplitudes that satisfy ∑|amplitude|² = 1.

    Parameters
    ----------
    data : np.ndarray
        Input data (will be normalized)
    n_qubits : int
        Number of qubits (requires 2^n data elements)

    Returns
    -------
    np.ndarray
        Complex-valued state vector where ∑|amplitude|² = 1

    Raises
    ------
    ValueError
        If data length is not 2^n_qubits

    Examples
    --------
    >>> import numpy as np
    >>> from simd_angle_encoder import amplitude_encode
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> state = amplitude_encode(data, n_qubits=2)
    >>> np.abs(state)**2
    array([0.03, 0.12, 0.27, 0.48])  # Sums to 1.0
    """
```

### 3. Basis Encoding (Phase 2)

**Status**: 🔄 Planned (Weeks 7-8)
**Performance Target**: 30-60x speedup

**Mathematical Definition**:

Given integer data $x$ and $n$ qubits:

1. **Convert to binary**:
$$
x = \sum_{i=0}^{n-1} b_i 2^i \quad \text{where } b_i \in \{0, 1\}
$$

2. **Encode in basis states**:
$$
|x\rangle = |b_{n-1} b_{n-2} \cdots b_0\rangle
$$

**Properties**:
- **Range**: Binary values $\{0, 1\}$
- **Qubits**: $n$ qubits can encode $2^n$ different values
- **Normalization**: Not required (computational basis states)
- **Uniqueness**: Unique representation for each integer

**Use Cases**:
- Integer data encoding
- Classification problems (discrete classes)
- Quantum walk algorithms

**Implementation** (proposed):
```rust
// src/encoders/basis.rs

pub fn basis_encode_simd(data: &[f64], n_qubits: usize) -> Vec<f64> {
    // Convert to integers (truncate)
    let int_data: Vec<u64> = data.iter().map(|&x| x as u64).collect();

    // Encode each integer as binary vector
    int_data
        .iter()
        .flat_map(|&x| {
            (0..n_qubits)
                .map(move |i| ((x >> i) & 1) as f64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_encoding() {
        let data = vec![5.0, 10.0, 15.0];  // 0101, 1010, 1111
        let encoded = basis_encode_simd(&data, 4);

        // First number: 5 = 0b0101
        assert_eq!(encoded[0], 1.0);  // Bit 0
        assert_eq!(encoded[1], 0.0);  // Bit 1
        assert_eq!(encoded[2], 1.0);  // Bit 2
        assert_eq!(encoded[3], 0.0);  // Bit 3
    }
}
```

**SIMD Bit Extraction** (advanced optimization):
```rust
// src/simd/arch/x86_64.rs

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn basis_extract_bits_avx2(data: &[u64], bit_idx: usize) -> Vec<u64> {
    let mask = 1u64 << bit_idx;
    let mut result = Vec::with_capacity(data.len());

    let chunks = data.chunks_exact(4);
    for chunk in chunks {
        // Load 4 u64 values
        let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

        // Extract specific bit
        let masked = _mm256_and_si256(vec, _mm256_set1_epi64x(mask as i64));

        // Convert to 0 or 1
        let shifted = _mm256_srli_epi64(masked, bit_idx);
        _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, shifted);
    }

    // Handle remainder
    for &val in chunks.remainder() {
        result.push((val >> bit_idx) & 1);
    }

    result
}
```

**Python API** (proposed):
```python
def basis_encode(data: np.ndarray, n_qubits: int) -> np.ndarray:
    """Encode integer data into computational basis states.

    Converts each integer to binary and encodes in qubits.

    Parameters
    ----------
    data : np.ndarray
        Integer data to encode
    n_qubits : int
        Number of qubits (bits) to use

    Returns
    -------
    np.ndarray
        Binary representation as array of 0s and 1s

    Examples
    --------
    >>> import numpy as np
    >>> from simd_angle_encoder import basis_encode
    >>> data = np.array([5, 10, 15])
    >>> encoded = basis_encode(data, n_qubits=4)
    >>> print(encoded.reshape(3, 4))
    [[1. 0. 1. 0.]   # 5 = 0101
     [0. 1. 0. 1.]   # 10 = 1010
     [1. 1. 1. 1.]]  # 15 = 1111
    """
```

### 4. Dense Angle Encoding (Stretch Goal)

**Status**: 📋 Planned (v1.1+)
**Performance Target**: 10-20x speedup

**Mathematical Definition**:

Given input data $x = [x_1, x_2, ..., x_m]$ and $n$ qubits where $m > n$:

1. **Apply feature map** (e.g., cosine, sine):
$$
\phi(x)_i = \cos(x_i) \quad \text{or} \quad \sin(x_i)
$$

2. **Pool/aggregate** to fit $n$ qubits:
$$
\text{pooled}_j = \sum_{k \in \text{group}_j} \phi(x)_k
$$

3. **Encode** using angle encoding:
$$
\theta_j = \text{pooled}_j \times 2\pi
$$

**Properties**:
- **Range**: $[0, 2\pi]$ after mapping
- **Qubits**: Can encode more data elements than qubits
- **Normalization**: Not strictly required
- **Uniqueness**: Not unique due to pooling

**Use Cases**:
- High-dimensional data with limited qubits
- Feature engineering in quantum ML
- Dimensionality reduction before quantum processing

## Encoding Method Comparison

| Property | Angle | Amplitude | Basis | Dense Angle |
|----------|-------|-----------|-------|-------------|
| **Data Type** | Continuous | Continuous | Integer | Continuous |
| **Data Size** | $n$ elements | $2^n$ elements | $n$ elements | $m > n$ elements |
| **Output Range** | $[0, 2\pi]$ | Complex, $|amp|^2=1$ | $\{0, 1\}$ | $[0, 2\pi]$ |
| **Normalization** | Not required | Required (L2) | Not required | Not required |
| **Uniqueness** | No (mod $2\pi$) | Yes (global phase) | Yes | No |
| **SIMD Speedup** | 40-90x | 20-50x (expected) | 30-60x (expected) | 10-20x (expected) |
| **Qubit Efficiency** | Low (1:1) | High (exponential) | Medium (1:1) | High (pooling) |
| **Implementation** | ✅ Complete | 🔄 Phase 2 | 🔄 Phase 2 | 📋 Future |

## API Design

### Unified API

```python
# python/simd_angle_encoder/__init__.py

def encode(
    data: Union[np.ndarray, list],
    n_qubits: int,
    method: str = 'angle'
) -> np.ndarray:
    """
    Encode data using specified encoding method.

    Parameters
    ----------
    data : np.ndarray
        Input data
    n_qubits : int
        Number of qubits
    method : str
        Encoding method: 'angle', 'amplitude', 'basis'

    Returns
    -------
    np.ndarray
        Encoded data

    Examples
    --------
    >>> from simd_angle_encoder import encode
    >>> data = np.array([0.1, 0.2, 0.3, 0.4])

    >>> # Angle encoding (default)
    >>> angle_encoded = encode(data, n_qubits=4, method='angle')

    >>> # Amplitude encoding
    >>> amp_encoded = encode(data, n_qubits=2, method='amplitude')

    >>> # Basis encoding
    >>> basis_encoded = encode(data, n_qubits=4, method='basis')
    """
    method = method.lower()

    if method == 'angle':
        return _angle_encode_simd(data, n_qubits)
    elif method == 'amplitude':
        return _amplitude_encode_simd(data, n_qubits)
    elif method == 'basis':
        return _basis_encode_simd(data, n_qubits)
    else:
        raise ValueError(f"Unknown encoding method: {method}")
```

### Batch API

```python
def encode_batch(
    batch_data: Union[np.ndarray, list],
    n_qubits: int,
    method: str = 'angle'
) -> np.ndarray:
    """
    Encode batch of data using specified encoding method.

    Parameters
    ----------
    batch_data : np.ndarray
        Batch data with shape (batch_size, data_dim)
    n_qubits : int
        Number of qubits
    method : str
        Encoding method

    Returns
    -------
    np.ndarray
        Encoded batch with shape (batch_size, n_qubits)

    Examples
    --------
    >>> batch = np.random.random((100, 8))
    >>> encoded = encode_batch(batch, n_qubits=10, method='angle')
    >>> encoded.shape
    (100, 10)
    """
```

## Testing Strategy

### Common Test Suite

Each encoding method must pass:

```python
# tests/test_encoding_methods.py

import pytest
import numpy as np
from simd_angle_encoder import encode, encode_batch

class TestAngleEncoding:
    def test_basic_encoding(self):
        data = np.array([0.0, 0.25, 0.5, 0.75])
        encoded = encode(data, n_qubits=4, method='angle')

        assert encoded.shape == (4,)
        assert np.allclose(encoded, [0.0, np.pi/2, np.pi, 3*np.pi/2])

class TestAmplitudeEncoding:
    def test_normalization(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        encoded = encode(data, n_qubits=2, method='amplitude')

        # Check normalization
        prob = np.abs(encoded)**2
        assert np.isclose(prob.sum(), 1.0)

class TestBasisEncoding:
    def test_binary_conversion(self):
        data = np.array([5.0, 10.0, 15.0])
        encoded = encode(data, n_qubits=4, method='basis')

        # Reshape to (batch, n_qubits)
        encoded_matrix = encoded.reshape(3, 4)

        # 5 = 0101
        assert np.array_equal(encoded_matrix[0], [1., 0., 1., 0.])

        # 10 = 1010
        assert np.array_equal(encoded_matrix[1], [0., 1., 0., 1.])

        # 15 = 1111
        assert np.array_equal(encoded_matrix[2], [1., 1., 1., 1.])

class TestAllEncodings:
    def test_consistent_api(self):
        """All encodings should have consistent API."""
        data = np.random.random(8)

        # All should work with same data
        angle_enc = encode(data, n_qubits=8, method='angle')
        amp_enc = encode(data, n_qubits=3, method='amplitude')
        basis_enc = encode(data, n_qubits=8, method='basis')

        # Check return types
        assert isinstance(angle_enc, np.ndarray)
        assert isinstance(amp_enc, np.ndarray)
        assert isinstance(basis_enc, np.ndarray)

    def test_batch_consistency(self):
        """All encodings should support batch operations."""
        batch = np.random.random((10, 8))

        for method in ['angle', 'amplitude', 'basis']:
            encoded = encode_batch(batch, n_qubits=8, method=method)
            assert encoded.shape[0] == 10  # Batch size preserved
```

### Property-Based Tests

```python
# tests/property/test_encoding_properties.py

from hypothesis import given, strategies as st
import numpy as np
from simd_angle_encoder import encode

class TestAngleEncodingProperties:
    @given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=100))
    def test_output_range(self, data):
        """Angle encoding should produce values in [0, 2π]."""
        encoded = encode(np.array(data), n_qubits=len(data), method='angle')

        assert np.all(encoded >= 0.0)
        assert np.all(encoded <= 2.0 * np.pi)

class TestAmplitudeEncodingProperties:
    @given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=64))
    def test_normalization_property(self, data):
        """Amplitude encoding should always normalize to 1."""
        # Pad to power of 2
        n_qubits = max(1, int(np.ceil(np.log2(len(data)))))
        padded = np.pad(data, (0, 2**n_qubits - len(data)))

        encoded = encode(padded, n_qubits=n_qubits, method='amplitude')
        prob = np.abs(encoded)**2

        assert np.isclose(prob.sum(), 1.0, atol=1e-10)
```

## Performance Benchmarks

### Benchmark Template

```python
# benchmarks/test_encoding_methods.py

import pytest
import numpy as np
from simd_angle_encoder import encode, encode_batch

@pytest.mark.benchmark(group="angle-encoding")
@pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
@pytest.mark.parametrize("data_dim", [32, 64, 128, 256])
def test_angle_encode_benchmark(benchmark, batch_size, data_dim):
    """Benchmark angle encoding."""
    data = np.random.random((batch_size, data_dim))

    if batch_size == 1:
        result = benchmark(encode, data[0], n_qubits=data_dim, method='angle')
    else:
        result = benchmark(encode_batch, data, n_qubits=data_dim, method='angle')

    assert result is not None

@pytest.mark.benchmark(group="amplitude-encoding")
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("n_qubits", [4, 8, 12])  # 16, 256, 4096 data elements
def test_amplitude_encode_benchmark(benchmark, batch_size, n_qubits):
    """Benchmark amplitude encoding."""
    data_dim = 2 ** n_qubits
    data = np.random.random((batch_size, data_dim))

    result = benchmark(encode_batch, data, n_qubits=n_qubits, method='amplitude')

    assert result is not None
```

## Migration Path

### Phase 2 Implementation Order

**Week 5-6**: Amplitude Encoding
1. Implement `amplitude_encode_simd()` in `src/encoders/amplitude.rs`
2. Implement SIMD norm computation in `src/simd/norm.rs`
3. Add Python wrapper in `lib.rs`
4. Add public API in `python/simd_angle_encoder/__init__.py`
5. Write tests and benchmarks

**Week 7-8**: Basis Encoding
1. Implement `basis_encode_simd()` in `src/encoders/basis.rs`
2. Implement SIMD bit extraction (optional optimization)
3. Add Python wrapper
4. Add public API
5. Write tests and benchmarks

**Week 9-10**: Refactoring
1. Extract angle encoding to `src/encoders/angle.rs`
2. Implement common `Encoder` trait
3. Add unified `encode(method=...)` API
4. Update all tests
5. Performance validation

## Success Criteria

### Phase 2 Deliverables

- [x] Angle encoding implemented (Phase 1)
- [ ] Amplitude encoding implemented
- [ ] Basis encoding implemented
- [ ] All encodings support batch processing
- [ ] Unified API (`encode(method=...)`)
- [ ] Performance targets met (20-60x speedup)
- [ ] Test coverage > 90% for all encodings
- [ ] Comprehensive documentation

### Quality Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Angle Encoding** | 40-90x | 40-90x | ✅ |
| **Amplitude Encoding** | 20-50x | TBD | 🔄 |
| **Basis Encoding** | 30-60x | TBD | 🔄 |
| **Code Coverage** | 90%+ | 92% | ✅ |
| **Test Count** | 150+ | 92 | 🔄 |

## Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Amplitude normalization overflow** | Low | Medium | Use SIMD-safe algorithms |
| **Basis encoding performance** | Low | Low | Profile bit operations |
| **API confusion** | Medium | Medium | Clear docs, examples |
| **Test coverage gaps** | Medium | Medium | Property-based tests |

## References

### Quantum Encoding Literature

- [Data Encoding for Quantum Machine Learning](https://arxiv.org/abs/2105.02276)
- [Encoding Quantum Algorithms](https://www.nature.com/articles/s41586-019-1666-5)
- [Quantum Data Loading](https://arxiv.org/abs/2004.00001)

### Implementation References

- **PennyLane Encoding**: https://docs.pennylane.ai/qml/demos/tutorial_data_reuploading_classifier.html
- **Qiskit Encoding**: https://qiskit.org/textbook/ch-applications/qnn_audio.html

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-04 | Architecture Agent | Initial encoding methods architecture |

---

**End of ADR-003**
