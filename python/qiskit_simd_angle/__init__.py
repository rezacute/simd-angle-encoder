"""
Qiskit SIMD Angle Encoder Plugin

SIMD-accelerated angle encoding for Qiskit quantum circuits.

Example:
    >>> from qiskit import QuantumCircuit
    >>> from qiskit_simd_angle import SIMDAngleEncoding
    >>>
    >>> qc = QuantumCircuit(4)
    >>> encoding = SIMDAngleEncoding([0.1, 0.2, 0.3, 0.4], num_qubits=4)
    >>> qc.compose(encoding, inplace=True)
"""

from qiskit_simd_angle.angle import (
    SIMDAngleEncoding,
    simd_angle_encoding,
)

__all__ = ["SIMDAngleEncoding", "simd_angle_encoding"]
__version__ = "0.1.0"
