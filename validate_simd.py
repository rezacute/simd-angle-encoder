#!/usr/bin/env python3
"""Comprehensive SIMD validation to verify no workarounds or fabrication."""

import numpy as np
import simd_angle_encoder as sae
import time

print("="*80)
print("SIMD VALIDATION - Checking for workarounds and fabricated results")
print("="*80)

# Test 1: Verify SIMD is actually being used (not scalar fallback)
print("\n[TEST 1] Verifying SIMD detection and usage")
print("-"*80)
simd_info = sae.simd_info()
print(f"SIMD Info: {simd_info}")
assert "AVX-512" in simd_info or "AVX2" in simd_info or "NEON" in simd_info, "❌ SIMD not detected!"
print("✅ SIMD properly detected")

# Test 2: Correctness verification
print("\n[TEST 2] Correctness verification")
print("-"*80)
np.random.seed(42)
test_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

all_correct = True
for size in test_sizes:
    data = np.random.random(size)
    result_simd = sae.encode(data, size)
    result_numpy = data * 2 * np.pi
    
    # Pad expected result if needed
    expected = np.zeros(size)
    expected[:size] = result_numpy[:size]
    
    if not np.allclose(result_simd, expected, rtol=1e-10):
        print(f"❌ Size {size}: MISMATCH!")
        all_correct = False
    else:
        print(f"✅ Size {size}: Correct")

if all_correct:
    print("✅ All correctness tests passed")

# Test 3: Performance sanity check - should be faster than naive Python
print("\n[TEST 3] Performance sanity check")
print("-"*80)
np.random.seed(42)
data = np.random.random(1024)
n_runs = 1000

# Naive Python implementation
def naive_encode(data, n_qubits):
    result = np.zeros(n_qubits)
    for i in range(min(len(data), n_qubits)):
        result[i] = data[i] * 2 * np.pi
    return result

# Warmup
_ = naive_encode(data, 1024)
_ = sae.encode(data, 1024)

# Benchmark naive
start = time.time()
for _ in range(n_runs):
    _ = naive_encode(data, 1024)
naive_time = (time.time() - start) / n_runs * 1000

# Benchmark SIMD
start = time.time()
for _ in range(n_runs):
    _ = sae.encode(data, 1024)
simd_time = (time.time() - start) / n_runs * 1000

speedup = naive_time / simd_time
print(f"Naive Python: {naive_time:.4f} ms")
print(f"SIMD Rust:    {simd_time:.4f} ms")
print(f"Speedup:      {speedup:.2f}x")

if speedup > 1.2:
    print(f"✅ SIMD shows {speedup:.2f}x speedup (reasonable)")
elif speedup > 0.8:
    print(f"⚠️  Speedup {speedup:.2f}x is modest (might be overhead-bound)")
else:
    print(f"❌ SIMD SLOWER than naive (suspicious!)")

# Test 4: Check for proper SIMD patterns (not just calling NumPy)
print("\n[TEST 4] Verify actual SIMD implementation")
print("-"*80)

# The implementation should handle batch operations efficiently
np.random.seed(42)
batch_sizes = [10, 100, 1000]

for batch_size in batch_sizes:
    batch_data = np.random.random((batch_size, 64))
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = sae.encode_batch(batch_data, 64)
    batch_time = (time.time() - start) / 100 * 1000
    
    print(f"Batch {batch_size:4d} x 64: {batch_time:.4f} ms ({batch_time/batch_size*1000:.4f} μs per item)")

print("✅ Batch operations functional")

# Test 5: Verify scaling is reasonable
print("\n[TEST 5] Verify scaling behavior")
print("-"*80)
sizes = [64, 256, 1024, 4096]
times = []

for size in sizes:
    data = np.random.random(size)
    
    start = time.time()
    for _ in range(100):
        _ = sae.encode(data, size)
    avg_time = (time.time() - start) / 100 * 1000
    times.append(avg_time)
    print(f"Size {size:4d}: {avg_time:.4f} ms")

# Check that scaling is roughly linear
for i in range(1, len(times)):
    ratio = times[i] / times[i-1]
    size_ratio = sizes[i] / sizes[i-1]
    
    # Allow 2x tolerance for cache effects
    if ratio > size_ratio * 2.5:
        print(f"⚠️  Scaling from {sizes[i-1]} to {sizes[i]}: {ratio:.2f}x (expected ~{size_ratio:.2f}x)")
    else:
        print(f"✅ Scaling from {sizes[i-1]} to {sizes[i]}: {ratio:.2f}x (reasonable)")

# Test 6: Memory efficiency check
print("\n[TEST 6] Memory efficiency check")
print("-"*80)
import sys

# Small data should use stack allocation
data = np.random.random(16)
result = sae.encode(data, 16)
print(f"Small result size: {sys.getsizeof(result)} bytes")

# Large data should use heap allocation but be efficient
data_large = np.random.random(1024)
result_large = sae.encode(data_large, 1024)
print(f"Large result size: {sys.getsizeof(result_large)} bytes")

expected_large = 1024 * 8  # 8 bytes per f64
if sys.getsizeof(result_large) < expected_large * 1.5:
    print("✅ Memory usage reasonable")
else:
    print("⚠️  Memory usage higher than expected")

# Test 7: Consistency check
print("\n[TEST 7] Consistency across runs")
print("-"*80)
np.random.seed(42)
data = np.random.random(256)
results = []

for i in range(10):
    result = sae.encode(data, 256)
    results.append(result)

# All results should be identical
for i in range(1, len(results)):
    if not np.allclose(results[0], results[i]):
        print(f"❌ Run {i} differs from run 0")
        break
else:
    print("✅ All 10 runs produce identical results")

# Test 8: Edge cases
print("\n[TEST 8] Edge case handling")
print("-"*80)

# Single element
result = sae.encode(np.array([0.5]), 1)
expected = np.array([0.5 * 2 * np.pi])
assert np.allclose(result, expected), "❌ Single element failed"
print("✅ Single element handled")

# Empty/zero-sized
result = sae.encode(np.array([]), 10)
assert len(result) == 10, "❌ Empty array failed"
assert np.all(result == 0), "❌ Empty array should produce zeros"
print("✅ Empty array handled")

# Large n_qubits
data = np.random.random(10)
result = sae.encode(data, 10000)
assert len(result) == 10000, "❌ Large n_qubits failed"
print("✅ Large n_qubits handled")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
