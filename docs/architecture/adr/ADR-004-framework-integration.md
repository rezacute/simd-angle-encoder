# ADR-004: Framework Integration Patterns

**Status**: Proposed
**Date**: 2026-01-04
**Deciders**: Architecture Agent, Integration Agent
**Related**: ADR-001 (Architecture Overview), ROADMAP_V1.md (Phase 3)

## Context

Quantum machine learning practitioners use specific frameworks (PennyLane, Qiskit, Cirq) for their work. A standalone encoding library is less valuable than one that integrates natively with these frameworks. Native integration provides:

1. **Seamless UX**: Framework users don't need to learn new APIs
2. **Gradient Support**: Automatic differentiation through framework's autograd
3. **Device Abstraction**: Works with simulators and real quantum hardware
4. **Type Safety**: Framework-specific types (pennylane.numpy, qiskit.QuantumCircuit)

However, each framework has different APIs, semantics, and design philosophies, creating integration challenges.

## Decision

We adopt a **separate package strategy** with framework-specific integrations:

```
simd-angle-encoder/                # Core package
├── src/lib.rs                     # Rust core
├── python/simd_angle_encoder/     # Python API
└── ...

pennylane-simd-angle-encoder/      # PennyLane integration (Phase 3, Week 11-12)
├── pennylane_simd_angle/
│   ├── __init__.py
│   ├── angle.py
│   ├── amplitude.py
│   └── basis.py
└── tests/

qiskit-simd-angle-encoder/         # Qiskit integration (Phase 3, Week 13-14)
├── qiskit_simd_angle/
│   ├── __init__.py
│   ├── encoders/
│   └── circuits/
└── tests/
```

### Integration Strategy Principles

1. **Separate Packages**: One package per framework (avoids dependency bloat)
2. **Native APIs**: Framework-specific operations/classes
3. **Shared Core**: All integrations use same Rust backend
4. **Minimal Overhead**: <10% overhead vs direct calls

## PennyLane Integration (Week 11-12)

### Design Philosophy

PennyLane uses **operations** as building blocks. Integration should provide PennyLane operations that call SIMD backend internally.

### Architecture

```
┌────────────────────────────────────────────────────────┐
│  PennyLane User Code                                   │
│  @qml.qnode(dev)                                       │
│  def circuit(data):                                    │
│      AngleEncoding(data, wires=range(4))  # Native op! │
│      qml.CNOT(wires=[0, 1])                            │
│      return qml.expval(qml.PauliZ(0))                  │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  PennyLane Operation (Our Integration)                │
│  class AngleEncoding(qml.operation.Operation):         │
│      def __init__(self, data, wires):                  │
│          self.angles = encode(data)  # SIMD backend    │
│      def decomposition(self):                          │
│          return [qml.RY(angle, wire) ...]             │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  Core SIMD Library (Rust)                              │
│  simd_angle_encode(data, n_qubits)                     │
└────────────────────────────────────────────────────────┘
```

### Implementation

**Package Structure**:
```
pennylane-simd-angle-encoder/
├── pennylane_simd_angle/
│   ├── __init__.py
│   ├── angle.py              # Angle encoding operation
│   ├── amplitude.py          # Amplitude encoding operation
│   ├── basis.py              # Basis encoding operation
│   └── _utils.py             # Helper functions
├── setup.py
├── setup.cfg
├── pyproject.toml
└── tests/
    ├── test_angle.py
    ├── test_amplitude.py
    ├── test_basis.py
    └── test_integration.py
```

**Angle Encoding Operation**:

```python
# pennylane_simd_angle/angle.py

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from simd_angle_encoder import encode, encode_batch

class AngleEncoding(qml.operation.Operation):
    """SIMD-accelerated angle encoding.

    This operation encodes classical data into rotation angles using
    SIMD-optimized Rust backend, then applies RY rotations to qubits.

    Args:
        data (array-like): Input data to encode. Must be 1D and length
            must match number of wires.
        wires (Iterable): Wires to apply encoding to.
        do_queue (bool): Whether to queue operation (default: True).

    Example:
        >>> dev = qml.device('default.qubit', wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(data):
        ...     AngleEncoding(data, wires=range(4))
        ...     return qml.state()
        >>> result = circuit([0.1, 0.2, 0.3, 0.4])
    """

    num_params = 1
    num_wires = qml.operation.WiresEnum.All
    grad_method = "A"
    grad_recipe = None

    def __init__(self, data, wires, do_queue=True):
        # Validate and convert data
        data = pnp.asarray(data, dtype=pnp.float64)

        if data.ndim != 1:
            raise ValueError(f"Data must be 1D, got shape {data.shape}")

        if len(data) != len(wires):
            raise ValueError(
                f"Data length ({len(data)}) must match "
                f"number of wires ({len(wires)})"
            )

        # Encode using SIMD backend
        # Note: Convert to numpy for Rust backend
        encoded = encode(data.numpy(), len(wires))
        self.angles = pnp.array(encoded, like=data)

        super().__init__(angles=self.angles, wires=wires, do_queue=do_queue)

    @property
    def num_params(self):
        return 1

    def decomposition(self):
        """Decompose into RY rotations.

        This allows PennyLane to use the operation with any device
        that supports RY gates.
        """
        return [
            qml.RY(self.angles[i], wires=self.wires[i])
            for i in range(len(self.wires))
        ]

    def adjoint(self):
        """Adjoint (inverse) operation.

        Negates all angles for inverse transformation.
        """
        return AngleEncoding(-self.angles, wires=self.wires)

    def adjoint_class(self):
        return AngleEncoding
```

**Amplitude Encoding Operation**:

```python
# pennylane_simd_angle/amplitude.py

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from simd_angle_encoder import amplitude_encode

class AmplitudeEncoding(qml.operation.Operation):
    """SIMD-accelerated amplitude encoding.

    Encodes classical data into quantum state amplitudes.
    Uses QubitStateVector initialization.

    Args:
        data (array-like): Input data to encode. Length must be 2^n_qubits.
        wires (Iterable): Wires to initialize.
        do_queue (bool): Whether to queue operation (default: True).

    Example:
        >>> dev = qml.device('default.qubit', wires=2)
        >>> @qml.qnode(dev)
        ... def circuit(data):
        ...     AmplitudeEncoding(data, wires=range(2))
        ...     return qml.state()
        >>> result = circuit([0.1, 0.2, 0.3, 0.4])
    """

    num_params = 1
    num_wires = qml.operation.WiresEnum.All
    grad_method = None  # State initialization not differentiable

    def __init__(self, data, wires, do_queue=True):
        data = pnp.asarray(data, dtype=pnp.float64)

        n_qubits = len(wires)
        expected_len = 2 ** n_qubits

        if len(data) != expected_len:
            raise ValueError(
                f"Data length ({len(data)}) must be 2^n_qubits = {expected_len}"
            )

        # Encode using SIMD backend (normalizes automatically)
        encoded = amplitude_encode(data.numpy())

        # Convert to complex state vector
        self.state = pnp.array(
            encoded + 0j,  # Add imaginary part (all zeros)
            like=data
        )

        super().__init__(state=self.state, wires=wires, do_queue=do_queue)

    @property
    def num_params(self):
        return 1

    def decomposition(self):
        """Decompose using QubitStateVector.

        This initializes the quantum state directly.
        """
        return [qml.QubitStateVector(self.state, wires=self.wires)]
```

**Basis Encoding Operation**:

```python
# pennylane_simd_angle/basis.py

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from simd_angle_encoder import basis_encode

class BasisEncoding(qml.operation.Operation):
    """SIMD-accelerated basis encoding.

    Encodes integer data into computational basis states.
    Uses X gates to set qubits to |1⟩ where bits are 1.

    Args:
        data (array-like): Integer data to encode.
        wires (Iterable): Wires to encode to.
        do_queue (bool): Whether to queue operation (default: True).

    Example:
        >>> dev = qml.device('default.qubit', wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(data):
        ...     BasisEncoding(data, wires=range(4))
        ...     return qml.state()
        >>> # 5 = 0b0101 → qubits 0 and 2 in |1⟩ state
        >>> result = circuit([5])
    """

    num_params = 1
    num_wires = qml.operation.WiresEnum.All
    grad_method = None  # Not differentiable

    def __init__(self, data, wires, do_queue=True):
        data = pnp.asarray(data, dtype=pnp.float64)

        # Encode using SIMD backend (returns binary vector)
        encoded = basis_encode(data.numpy(), len(wires))

        # Convert to boolean array
        self.bits = pnp.array(encoded > 0.5, like=data)

        super().__init__(bits=self.bits, wires=wires, do_queue=do_queue)

    @property
    def num_params(self):
        return 1

    def decomposition(self):
        """Decompose using X gates.

        Apply X gate to each qubit where bit is 1.
        """
        ops = []
        for i, bit in enumerate(self.bits):
            if bit:  # If bit is 1, apply X gate
                ops.append(qml.X(wires=self.wires[i]))
        return ops
```

**Batch Processing Support**:

```python
# pennylane_simd_angle/batch.py

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from simd_angle_encoder import encode_batch

class BatchAngleEncoding(qml.operation.Operation):
    """Batch angle encoding for multiple data vectors.

    Processes multiple data vectors in parallel using SIMD batching.

    Args:
        batch_data (array-like): Batch data with shape (batch_size, data_dim).
        wires (Iterable): Wires to apply encoding to.
        do_queue (bool): Whether to queue operation (default: True).

    Example:
        >>> dev = qml.device('default.qubit', wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(batch):
        ...     # Encode entire batch at once
        ...     angles = BatchAngleEncoding(batch, wires=range(4))
        ...     return angles  # Returns (batch_size, n_qubits) array
        >>> batch = np.random.random((10, 4))
        >>> result = circuit(batch)
        >>> result.shape
        (10, 4)
    """

    num_params = 1
    num_wires = qml.operation.WiresEnum.All
    grad_method = "F"  # Finite differences for batch

    def __init__(self, batch_data, wires, do_queue=True):
        batch_data = pnp.asarray(batch_data, dtype=pnp.float64)

        if batch_data.ndim != 2:
            raise ValueError(
                f"Batch data must be 2D, got shape {batch_data.shape}"
            )

        batch_size, data_dim = batch_data.shape

        if data_dim != len(wires):
            raise ValueError(
                f"Data dimension ({data_dim}) must match "
                f"number of wires ({len(wires)})"
            )

        # Encode batch using SIMD
        encoded = encode_batch(batch_data.numpy(), len(wires))

        # Convert to PennyLane array
        self.angles_batch = pnp.array(encoded, like=batch_data)

        super().__init__(
            angles_batch=self.angles_batch,
            wires=wires,
            do_queue=do_queue
        )

    def decomposition(self):
        """Return angles for manual processing.

        Note: PennyLane doesn't natively support batch operations,
        so this returns the angles for user to process.
        """
        return self.angles_batch
```

### Testing

```python
# tests/test_angle.py

import pytest
import pennylane as qml
import numpy as np
from pennylane_simd_angle import AngleEncoding

class TestAngleEncoding:
    def test_basic_encoding(self):
        """Test basic angle encoding in QNode."""
        dev = qml.device('default.qubit', wires=4)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(4))
            return qml.state()

        data = np.array([0.0, 0.25, 0.5, 0.75])
        result = circuit(data)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_gradient_computation(self):
        """Test gradient flows through encoding."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            qml.RY(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        data = np.array([0.1, 0.2], requires_grad=True)
        grad = qml.grad(circuit)(data)

        assert grad is not None
        assert grad.shape == data.shape

    def test_adjoint_operation(self):
        """Test adjoint (inverse) operation."""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(data):
            AngleEncoding(data, wires=range(2))
            qml.adjoint(AngleEncoding)(data, wires=range(2))
            return qml.state()

        data = np.array([0.1, 0.2])
        result = circuit(data)

        # Should return to |00⟩ state
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)
```

## Qiskit Integration (Week 13-14)

### Design Philosophy

Qiskit uses **QuantumCircuit** builder pattern. Integration should provide helper functions that create circuits with encoding.

### Architecture

```
┌────────────────────────────────────────────────────────┐
│  Qiskit User Code                                      │
│  from qiskit_simd_angle import angle_encode_circuit    │
│  data = np.array([0.1, 0.2, 0.3, 0.4])                │
│  qc = angle_encode_circuit(data, n_qubits=4)           │
│  qc.draw()                                             │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  Qiskit Circuit Builder (Our Integration)             │
│  def angle_encode_circuit(data, n_qubits):             │
│      angles = encode(data)  # SIMD backend             │
│      qc = QuantumCircuit(n_qubits)                     │
│      for i, angle in enumerate(angles):                │
│          qc.ry(angle, i)                               │
│      return qc                                         │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  Core SIMD Library (Rust)                              │
│  simd_angle_encode(data, n_qubits)                     │
└────────────────────────────────────────────────────────┘
```

### Implementation

**Package Structure**:
```
qiskit-simd-angle-encoder/
├── qiskit_simd_angle/
│   ├── __init__.py
│   ├── encoders/
│   │   ├── angle.py
│   │   ├── amplitude.py
│   │   └── basis.py
│   ├── circuits/
│   │   ├── builder.py
│   │   └── batch.py
│   └── utils.py
├── setup.py
├── setup.cfg
└── tests/
    ├── test_angle.py
    ├── test_amplitude.py
    └── test_batch.py
```

**Angle Encoding Circuit Builder**:

```python
# qiskit_simd_angle/encoders/angle.py

from qiskit import QuantumCircuit
import numpy as np
from simd_angle_encoder import encode

def angle_encode_circuit(data, n_qubits):
    """Create Qiskit circuit with angle encoding.

    Args:
        data (np.ndarray): Input data to encode.
        n_qubits (int): Number of qubits.

    Returns:
        QuantumCircuit: Circuit with encoded rotations.

    Example:
        >>> data = np.array([0.1, 0.2, 0.3, 0.4])
        >>> qc = angle_encode_circuit(data, n_qubits=4)
        >>> qc.draw()
             ┌───┐
        q_0: ┤ RY(0.5) ├──
             ├───┤
        q_1: ┤ RY(1.2) ├──
             ├───┤
        q_2: ┤ RY(3.1) ├──
             ├───┤
        q_3: ┤ RY(2.4) ├──
             └───┘
    """
    # Encode using SIMD
    angles = encode(data, n_qubits)

    # Create circuit
    qc = QuantumCircuit(n_qubits)

    # Add RY gates
    for i, angle in enumerate(angles):
        qc.ry(angle, i)

    return qc

def angle_encode_parameterized(n_qubits):
    """Create parameterized circuit for variational use.

    Returns:
        tuple: (QuantumCircuit, ParameterVector)

    Example:
        >>> qc, params = angle_encode_parameterized(4)
        >>> # Later bind parameters
        >>> data = np.random.random(4)
        >>> bound_qc = qc.bind_parameters({params[i]: value for i, value in enumerate(data)})
    """
    from qiskit.circuit import ParameterVector

    # Create parameters
    param_vector = ParameterVector('θ', n_qubits)

    # Create circuit
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(param_vector[i], i)

    return qc, param_vector
```

**Batch Circuit Builder**:

```python
# qiskit_simd_angle/circuits/batch.py

from qiskit import QuantumCircuit
import numpy as np
from simd_angle_encoder import encode_batch

class BatchEncoder:
    """Batch encode multiple data vectors into Qiskit circuits.

    Args:
        n_qubits (int): Number of qubits for encoding.

    Example:
        >>> encoder = BatchEncoder(n_qubits=4)
        >>> batch_data = np.random.random((10, 8))
        >>> circuits = encoder.encode_batch(batch_data)
        >>> len(circuits)
        10
    """

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def encode_batch(self, batch_data):
        """Encode batch of data into list of circuits.

        Args:
            batch_data (np.ndarray): Shape (batch_size, data_dim)

        Returns:
            list[QuantumCircuit]: List of circuits (one per data vector)
        """
        # Encode all data using SIMD
        encoded = encode_batch(batch_data, self.n_qubits)

        # Create circuits
        circuits = []
        for i in range(batch_data.shape[0]):
            qc = QuantumCircuit(self.n_qubits)
            for j in range(self.n_qubits):
                qc.ry(encoded[i, j], j)
            circuits.append(qc)

        return circuits

    def encode_batch_transpiled(self, batch_data, backend):
        """Encode and transpile for specific backend.

        Args:
            batch_data (np.ndarray): Input data
            backend (Backend): Qiskit backend

        Returns:
            list[QuantumCircuit]: Transpiled circuits
        """
        circuits = self.encode_batch(batch_data)
        from qiskit import transpile
        return transpile(circuits, backend=backend)
```

**Amplitude Encoding**:

```python
# qiskit_simd_angle/encoders/amplitude.py

from qiskit import QuantumCircuit
import numpy as np
from simd_angle_encoder import amplitude_encode

def amplitude_encode_circuit(data, n_qubits):
    """Create circuit with amplitude encoding initialization.

    Args:
        data (np.ndarray): Input data (length must be 2^n_qubits).
        n_qubits (int): Number of qubits.

    Returns:
        QuantumCircuit: Circuit with initialized state.

    Note:
        Uses `initialize` which decomposes to standard gates.
        For large n_qubits, decomposition may be expensive.
    """
    # Get amplitudes (normalizes automatically)
    amplitudes = amplitude_encode(data)

    # Create circuit
    qc = QuantumCircuit(n_qubits)

    # Initialize state
    qc.initialize(amplitudes, range(n_qubits))

    return qc
```

### Testing

```python
# tests/test_qiskit_integration.py

import pytest
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit_simd_angle import angle_encode_circuit, BatchEncoder

class TestQiskitIntegration:
    def test_angle_encode_circuit(self):
        """Test basic circuit creation."""
        data = np.random.random(4)
        qc = angle_encode_circuit(data, n_qubits=4)

        assert qc.num_qubits == 4
        assert qc.depth() == 1
        assert len(qc.data) == 4

    def test_circuit_execution(self):
        """Test circuit runs on simulator."""
        data = np.array([0.0, 0.5, 0.25, 0.75])
        qc = angle_encode_circuit(data, n_qubits=4)

        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        statevector = result.get_statevector()

        assert statevector is not None
        assert len(statevector) == 16

    def test_batch_encoding(self):
        """Test batch circuit creation."""
        batch_data = np.random.random((10, 4))
        encoder = BatchEncoder(n_qubits=4)
        circuits = encoder.encode_batch(batch_data)

        assert len(circuits) == 10
        for qc in circuits:
            assert qc.num_qubits == 4
```

## Comparison: Framework Integration Approaches

### Approach 1: Separate Packages (CHOSEN)

**Pros**:
- No dependency bloat (PennyLane users don't need Qiskit)
- Framework-specific optimizations
- Clear separation of concerns
- Independent versioning

**Cons**:
- More packages to maintain
- Code duplication (helper functions)

### Approach 2: Monolithic Package

**Pros**:
- Single package to install
- Shared code

**Cons**:
- Dependency bloat (all users get all frameworks)
- Versioning conflicts
- Larger install size

**Decision**: Separate packages (Approach 1) is better for users.

## Integration Quality Checklist

### PennyLane Integration

- [ ] Operation class for each encoding method
- [ ] Gradient computation support
- [ ] Adjoint operation support
- [ ] Batch processing support
- [ ] Documentation with examples
- [ ] Integration tests
- [ ] Performance benchmarks (30-40x vs built-in)

### Qiskit Integration

- [ ] Circuit builder for each encoding method
- [ ] Parameterized circuit support
- [ ] Batch processing support
- [ ] Transpilation support
- [ ] Documentation with examples
- [ ] Integration tests
- [ ] Performance benchmarks

## Performance Targets

| Framework | Encoding Method | Target vs Built-in | Measurement Method |
|-----------|----------------|-------------------|-------------------|
| PennyLane | Angle | 30-40x faster | Time per QNode execution |
| PennyLane | Amplitude | 20-30x faster | Time per QNode execution |
| PennyLane | Basis | 25-35x faster | Time per QNode execution |
| Qiskit | Angle | 30-40x faster | Circuit construction time |
| Qiskit | Amplitude | 20-30x faster | Circuit construction time |
| Qiskit | Basis | 25-35x faster | Circuit construction time |

## Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Framework API changes** | Low | Medium | Pin framework versions, monitor releases |
| **Integration overhead** | Medium | Low | Profile to keep <10% overhead |
| **Different semantics** | Medium | Medium | Clear documentation of differences |
| **Testing complexity** | Medium | Low | Comprehensive integration tests |

## References

### Framework Documentation

- **PennyLane**: https://docs.pennylane.ai/
- **Qiskit**: https://qiskit.org/documentation/

### Integration Examples

- PennyLane Operation API: https://docs.pennylane.ai/en/stable/dev/operation.html
- Qiskit Circuit API: https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-04 | Architecture Agent | Initial framework integration design |

---

**End of ADR-004**
