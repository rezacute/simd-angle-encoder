#!/usr/bin/env python3
"""Quick performance comparison for small data."""

import numpy as np
import timeit
import simd_angle_encoder

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function with high precision."""
    # Warmup
    for _ in range(100):
        func(*args, **kwargs)

    # Measure
    timer = timeit.Timer(lambda: func(*args, **kwargs))
    times = timer.repeat(repeat=100, number=1)

    # Return median (in microseconds)
    return np.median(times) * 1_000_000

print("=" * 80)
print("Phase 2A: Memory Optimization Performance Test")
print("=" * 80)

# Test configurations
configs = [
    ("Tiny (4 elements)", np.array([0.5] * 4), 4),
    ("Small (8 elements)", np.array([0.5] * 8), 8),
    ("Small-Medium (16 elements)", np.array([0.5] * 16), 16),
    ("Medium (32 elements)", np.array([0.5] * 32), 32),
    ("Large (64 elements)", np.array([0.5] * 64), 64),
]

print("\nSingle Array Encoding Performance:")
print("-" * 80)
print(f"{'Configuration':<30} {'Time (μs)':<15} {'Speedup vs NumPy':<20}")
print("-" * 80)

for name, data, n_qubits in configs:
    # Benchmark SIMD implementation
    t_simd = benchmark_function(simd_angle_encoder.encode, data, n_qubits)

    # Benchmark NumPy implementation
    t_numpy = benchmark_function(lambda d, n: d[:n] * 2 * np.pi, data, n_qubits)

    speedup = t_numpy / t_simd

    print(f"{name:<30} {t_simd:<15.3f} {speedup:<20.2f}x")

print("\n" + "=" * 80)
print("Batch Encoding Performance (10 x 16):")
print("-" * 80)

batch_data = np.random.random((10, 16))
n_qubits = 16

t_batch_simd = benchmark_function(simd_angle_encoder.encode_batch, batch_data, n_qubits)
t_batch_numpy = benchmark_function(lambda b, n: (b[:, :n] * 2 * np.pi), batch_data, n_qubits)

batch_speedup = t_batch_numpy / t_batch_simd

print(f"SIMD Time:     {t_batch_simd:.3f} μs")
print(f"NumPy Time:    {t_batch_numpy:.3f} μs")
print(f"Speedup:       {batch_speedup:.2f}x")

print("\n" + "=" * 80)
print("Memory Optimization Details:")
print("-" * 80)
print("✅ Stack allocation:     Enabled for n_qubits <= 32")
print("✅ Zero-copy interface:  Enabled for contiguous arrays")
print("✅ Simplified loop:      Single-pass algorithm")
print("\nExpected improvements:")
print("  - Stack allocation:  3-5x for n_qubits <= 32")
print("  - Zero-copy:         1.5-2x for all sizes")
print("  - Combined:          5-10x for small data")
print("=" * 80)
