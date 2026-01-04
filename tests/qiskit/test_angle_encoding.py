"""
Tests for Qiskit SIMD Angle Encoding plugin.
"""

import sys
from pathlib import Path

# Add python directory to path for qiskit_simd_angle
sys.path.append(str(Path(__file__).parent.parent.parent / "python"))

import pytest
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_simd_angle import SIMDAngleEncoding, simd_angle_encoding
except ImportError:
    pytest.skip("Qiskit SIMD plugin not available", allow_module_level=True)


class TestSIMDAngleEncodingBasic:
    """Test basic angle encoding functionality."""

    def test_basic_encoding(self):
        """Test basic circuit creation."""
        data = [0.1, 0.2, 0.3, 0.4]
        circuit = SIMDAngleEncoding(data)
        
        assert circuit.num_qubits == 4
        assert circuit.name == "SIMDAngleEncoding"

    def test_custom_num_qubits(self):
        """Test encoding with more qubits than data."""
        data = [0.1, 0.2]
        circuit = SIMDAngleEncoding(data, num_qubits=4)
        
        assert circuit.num_qubits == 4

    def test_encoding_correctness(self):
        """Test that encoding produces correct RY angles."""
        data = np.array([0.0, 0.5, 1.0])
        circuit = SIMDAngleEncoding(data)
        
        # Check circuit has RY gates with correct angles
        expected = data * 2 * np.pi
        ry_angles = []
        for instr in circuit.data:
            if instr.operation.name == "ry":
                ry_angles.append(instr.operation.params[0])
        
        # Only non-zero angles create gates
        assert np.allclose(ry_angles, expected[expected != 0])

    def test_compose_with_circuit(self):
        """Test composing encoding with existing circuit."""
        qc = QuantumCircuit(4)
        qc.h(0)
        
        encoding = SIMDAngleEncoding([0.1, 0.2, 0.3, 0.4])
        qc.compose(encoding, inplace=True)
        
        assert qc.num_qubits == 4


class TestSIMDAngleEncodingFunction:
    """Test functional API."""

    def test_basic_function(self):
        """Test simd_angle_encoding function."""
        qc = QuantumCircuit(4)
        simd_angle_encoding(qc, [0.1, 0.2, 0.3, 0.4])
        
        assert qc.num_qubits == 4

    def test_custom_qubits(self):
        """Test encoding to specific qubits."""
        qc = QuantumCircuit(6)
        simd_angle_encoding(qc, [0.1, 0.2], qubits=[2, 4])
        
        # Check gates are on correct qubits
        for instr in qc.data:
            if instr.operation.name == "ry":
                qubit_idx = qc.find_bit(instr.qubits[0]).index
                assert qubit_idx in [2, 4]

    def test_returns_circuit(self):
        """Test that function returns the circuit."""
        qc = QuantumCircuit(2)
        result = simd_angle_encoding(qc, [0.1, 0.2])
        
        assert result is qc


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_data(self):
        """Test with empty data."""
        circuit = SIMDAngleEncoding([], num_qubits=2)
        assert circuit.num_qubits == 2

    def test_single_qubit(self):
        """Test single qubit encoding."""
        circuit = SIMDAngleEncoding([0.5])
        assert circuit.num_qubits == 1

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        data = np.array([0.1, 0.2, 0.3])
        circuit = SIMDAngleEncoding(data)
        assert circuit.num_qubits == 3

    def test_list_input(self):
        """Test with list input."""
        data = [0.1, 0.2, 0.3]
        circuit = SIMDAngleEncoding(data)
        assert circuit.num_qubits == 3
