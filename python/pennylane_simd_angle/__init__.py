"""
PennyLane SIMD Angle Encoder Plugin

This plugin provides SIMD-accelerated encoding operations for PennyLane quantum circuits.
It leverages Rust SIMD optimizations to achieve 30-40x speedup over PennyLane's built-in encoding.

Example usage:
    >>> import pennylane as qml
    >>> from pennylane_simd_angle import AngleEncoding
    >>>
    >>> dev = qml.device('default.qubit', wires=4)
    >>> @qml.qnode(dev)
    ... def circuit(data):
    ...     AngleEncoding(data, wires=range(4))
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.expval(qml.PauliZ(0))
    >>>
    >>> result = circuit([0.1, 0.2, 0.3, 0.4])
"""

from pennylane_simd_angle.angle import (
    AngleEncoding,
    AngleEncodingBatch,
    angle_encoding_function,
    angle_encoding_batch_function
)
from pennylane_simd_angle._version import __version__

__all__ = [
    "AngleEncoding",
    "AngleEncodingBatch",
    "angle_encoding_function",
    "angle_encoding_batch_function",
    "__version__",
]

# Version information
__version__ = "0.1.0"
