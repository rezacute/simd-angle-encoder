"""
SIMD-accelerated angle encoding for Qiskit circuits.

Provides high-performance angle encoding using Rust SIMD optimizations.
"""

import numpy as np
from typing import Union, Optional, List
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

try:
    from simd_angle_encoder import encode, encode_batch
except ImportError:
    raise ImportError(
        "simd_angle_encoder package not found. "
        "Please install it first: pip install simd-angle-encoder"
    )


class SIMDAngleEncoding(QuantumCircuit):
    """SIMD-accelerated angle encoding circuit for Qiskit.

    Encodes classical data into rotation angles using SIMD-optimized
    Rust backend, applying RY rotations to qubits.

    Args:
        data: Input data to encode (1D array-like).
        num_qubits: Number of qubits. Defaults to len(data).
        name: Circuit name.

    Example:
        >>> from qiskit_simd_angle import SIMDAngleEncoding
        >>> encoding = SIMDAngleEncoding([0.1, 0.2, 0.3, 0.4])
        >>> print(encoding)
    """

    def __init__(
        self,
        data: Union[np.ndarray, List[float]],
        num_qubits: Optional[int] = None,
        name: str = "SIMDAngleEncoding",
    ):
        data = np.asarray(data, dtype=np.float64).flatten()
        
        if num_qubits is None:
            num_qubits = len(data)

        super().__init__(num_qubits, name=name)

        # SIMD-accelerated encoding
        angles = encode(data, num_qubits)

        # Apply RY rotations
        for i, angle in enumerate(angles):
            if angle != 0.0:
                self.ry(angle, i)


def simd_angle_encoding(
    circuit: QuantumCircuit,
    data: Union[np.ndarray, List[float]],
    qubits: Optional[List[int]] = None,
) -> QuantumCircuit:
    """Apply SIMD angle encoding to an existing circuit.

    Args:
        circuit: Target quantum circuit.
        data: Input data to encode.
        qubits: Target qubits. Defaults to [0, 1, ..., len(data)-1].

    Returns:
        The modified circuit.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> from qiskit_simd_angle import simd_angle_encoding
        >>> qc = QuantumCircuit(4)
        >>> simd_angle_encoding(qc, [0.1, 0.2, 0.3, 0.4])
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    
    if qubits is None:
        qubits = list(range(len(data)))

    angles = encode(data, len(qubits))

    for i, qubit in enumerate(qubits):
        if angles[i] != 0.0:
            circuit.ry(angles[i], qubit)

    return circuit
