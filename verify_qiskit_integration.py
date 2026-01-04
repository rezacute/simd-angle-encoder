#!/usr/bin/env python3
"""
Comprehensive verification script for Qiskit SIMD Angle Encoding integration.

This script verifies that:
1. The Qiskit plugin correctly calls the SIMD backend
2. No fabrication of results (correct mathematical encoding)
3. Angle encoding follows the formula: angle = data_value * 2π
4. Circuit generation is correct
5. Batch and single encoding produce identical results
"""

import sys
import numpy as np
from pathlib import Path

# Add local python directory to path for the Qiskit plugin
# We append (not insert) so installed packages take precedence
python_dir = Path(__file__).parent / "python"
if python_dir.exists():
    sys.path.append(str(python_dir))

print("=" * 70)
print("Qiskit SIMD Angle Encoder - Comprehensive Verification")
print("=" * 70)

# Test 1: Verify core SIMD module loads
print("\n[Test 1] Loading core SIMD module...")
try:
    from simd_angle_encoder import encode, encode_batch, simd_info
    print("✓ Core SIMD module loaded successfully")
    print(f"  {simd_info()}")
except ImportError as e:
    print(f"✗ Failed to load core SIMD module: {e}")
    sys.exit(1)

# Test 2: Verify Qiskit plugin loads
print("\n[Test 2] Loading Qiskit plugin...")
try:
    from qiskit_simd_angle import SIMDAngleEncoding, simd_angle_encoding
    from qiskit import QuantumCircuit
    print("✓ Qiskit plugin loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load Qiskit plugin: {e}")
    sys.exit(1)

# Test 3: Verify mathematical correctness (angle = data * 2π)
print("\n[Test 3] Verifying mathematical correctness...")
# Use non-zero data to avoid skipped gates in circuit
test_data = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
expected_angles = test_data * 2 * np.pi

# Test core SIMD function
simd_result = encode(test_data, n_qubits=5)
if np.allclose(simd_result, expected_angles, rtol=1e-10):
    print("✓ Core SIMD encoding mathematically correct")
    print(f"  Input:    {test_data}")
    print(f"  Expected: {expected_angles}")
    print(f"  Got:      {simd_result}")
else:
    print("✗ SIMD encoding has mathematical errors")
    print(f"  Expected: {expected_angles}")
    print(f"  Got:      {simd_result}")
    print(f"  Max error: {np.max(np.abs(simd_result - expected_angles))}")
    sys.exit(1)

# Test 4: Verify Qiskit circuit generation
print("\n[Test 4] Verifying Qiskit circuit generation...")
circuit = SIMDAngleEncoding(test_data, num_qubits=5)

# Extract RY gate angles from circuit, indexed by qubit
ry_angles_by_qubit = {}
for instr in circuit.data:
    if instr.operation.name == "ry":
        qubit_idx = circuit.find_bit(instr.qubits[0]).index
        angle = instr.operation.params[0]
        ry_angles_by_qubit[qubit_idx] = angle

# Create array in qubit order
ry_angles = np.array([ry_angles_by_qubit.get(i, 0.0) for i in range(5)])

if np.allclose(ry_angles, expected_angles, rtol=1e-10):
    print("✓ Qiskit circuit has correct RY angles")
    print(f"  Number of RY gates: {len(ry_angles_by_qubit)}")
    print(f"  Angles: {ry_angles}")
else:
    print("✗ Qiskit circuit has incorrect RY angles")
    print(f"  Expected: {expected_angles}")
    print(f"  Got:      {ry_angles}")
    sys.exit(1)

# Test 5: Verify functional API
print("\n[Test 5] Verifying functional API...")
qc = QuantumCircuit(5)
result_qc = simd_angle_encoding(qc, test_data)

# Extract angles by qubit index
functional_angles_by_qubit = {}
for instr in result_qc.data:
    if instr.operation.name == "ry":
        qubit_idx = result_qc.find_bit(instr.qubits[0]).index
        angle = instr.operation.params[0]
        functional_angles_by_qubit[qubit_idx] = angle

functional_angles = np.array([functional_angles_by_qubit.get(i, 0.0) for i in range(5)])

if np.allclose(functional_angles, expected_angles, rtol=1e-10):
    print("✓ Functional API produces correct angles")
else:
    print("✗ Functional API has incorrect angles")
    print(f"  Expected: {expected_angles}")
    print(f"  Got:      {functional_angles}")
    sys.exit(1)

# Test 6: Verify batch encoding consistency
print("\n[Test 6] Verifying batch encoding consistency...")
single_results = []
batch_data = np.random.random((10, 5))

# Encode each sample individually
for i in range(10):
    single_result = encode(batch_data[i], n_qubits=5)
    single_results.append(single_result)

# Encode as batch
batch_result = encode_batch(batch_data, n_qubits=5)

single_results = np.array(single_results)
if np.allclose(single_results, batch_result, rtol=1e-10):
    print("✓ Batch encoding consistent with single encoding")
    print(f"  Tested {batch_data.shape[0]} samples with {batch_data.shape[1]} features")
else:
    print("✗ Batch encoding differs from single encoding")
    print(f"  Max difference: {np.max(np.abs(single_results - batch_result))}")
    sys.exit(1)

# Test 7: Verify edge cases
print("\n[Test 7] Verifying edge cases...")

# Zero data
zero_data = np.zeros(5)
zero_result = encode(zero_data, n_qubits=5)
if np.allclose(zero_result, np.zeros(5)):
    print("✓ Zero data handled correctly")
else:
    print("✗ Zero data not handled correctly")
    sys.exit(1)

# One data
ones_data = np.ones(5)
ones_result = encode(ones_data, n_qubits=5)
if np.allclose(ones_result, 2 * np.pi * np.ones(5)):
    print("✓ Ones data handled correctly")
else:
    print("✗ Ones data not handled correctly")
    sys.exit(1)

# Data larger than n_qubits
large_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
truncated_result = encode(large_data, n_qubits=3)
expected_truncated = large_data[:3] * 2 * np.pi
if np.allclose(truncated_result, expected_truncated):
    print("✓ Data truncation with n_qubits works correctly")
else:
    print("✗ Data truncation not handled correctly")
    sys.exit(1)

# Test 8: Verify no fabrication (deterministic behavior)
print("\n[Test 8] Verifying deterministic behavior (no fabrication)...")
test_data = np.array([0.123456789, 0.987654321, 0.555555555])

# Run encoding multiple times
results = []
for _ in range(5):
    result = encode(test_data, n_qubits=3)
    results.append(result)

# Check all results are identical
for i in range(1, len(results)):
    if not np.allclose(results[0], results[i], rtol=1e-14):
        print("✗ Non-deterministic behavior detected!")
        print(f"  Run 1: {results[0]}")
        print(f"  Run {i+1}: {results[i]}")
        sys.exit(1)

print("✓ Results are deterministic (no random fabrication)")

# Test 9: Verify with different qubit indices
print("\n[Test 9] Verifying custom qubit indices...")
qc = QuantumCircuit(6)
test_data = np.array([0.1, 0.2, 0.3])
custom_qubits = [1, 3, 5]

result_qc = simd_angle_encoding(qc, test_data, qubits=custom_qubits)

# Check gates are on correct qubits
qubit_indices = []
for instr in result_qc.data:
    if instr.operation.name == "ry":
        qubit_idx = result_qc.find_bit(instr.qubits[0]).index
        qubit_indices.append(qubit_idx)

if qubit_indices == custom_qubits:
    print("✓ Custom qubit indices work correctly")
    print(f"  Qubits used: {qubit_indices}")
else:
    print("✗ Custom qubit indices not working correctly")
    print(f"  Expected: {custom_qubits}")
    print(f"  Got:      {qubit_indices}")
    sys.exit(1)

# Test 10: Performance sanity check
print("\n[Test 10] Performance sanity check...")
import time

large_data = np.random.random(1000)
n_qubits = 100

# Measure SIMD performance
start = time.time()
simd_result = encode(large_data, n_qubits)
simd_time = (time.time() - start) * 1000  # ms

# Measure NumPy performance
start = time.time()
numpy_result = large_data[:n_qubits] * 2 * np.pi
numpy_time = (time.time() - start) * 1000  # ms

print(f"  SIMD time:  {simd_time:.4f} ms")
print(f"  NumPy time: {numpy_time:.4f} ms")
print(f"  Speedup:    {numpy_time/simd_time:.2f}x")

# Verify results match
if np.allclose(simd_result, numpy_result, rtol=1e-10):
    print("✓ Performance sanity check passed (results match)")
else:
    print("✗ Performance sanity check failed (results don't match)")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - ALL TESTS PASSED")
print("=" * 70)
print("\nSummary:")
print("  ✓ Core SIMD module loads and works correctly")
print("  ✓ Qiskit plugin integrates properly")
print("  ✓ Mathematical encoding is correct (angle = data * 2π)")
print("  ✓ Circuit generation produces correct RY gates")
print("  ✓ Batch encoding is consistent with single encoding")
print("  ✓ Edge cases handled properly")
print("  ✓ Results are deterministic (no fabrication)")
print("  ✓ Custom qubit indices work correctly")
print("  ✓ Performance sanity check passed")
print("\nThe Qiskit SIMD Angle Encoder integration is verified to be")
print("mathematically correct and does not fabricate results.")
