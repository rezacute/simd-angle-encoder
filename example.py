#!/usr/bin/env python3
"""
Example usage of the SIMD Angle Encoder
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from simd_angle_encoder import encode, encode_batch, benchmark, simd_info

def main():
    print("SIMD Angle Encoder Example")
    print("-" * 50)
    
    # Print SIMD information
    print(f"SIMD Information:\n{simd_info()}\n")
    
    # Example 1: Simple encoding
    print("Example 1: Single Vector Encoding")
    data = np.random.random(8)
    n_qubits = 10
    
    print(f"Input data: {data}")
    encoded = encode(data, n_qubits)
    print(f"Encoded angles: {encoded}")
    print("-" * 50)
    
    # Example 2: Batch encoding
    print("Example 2: Batch Encoding")
    batch_size = 3
    data_size = 8
    batch_data = np.random.random((batch_size, data_size))
    
    print(f"Input batch shape: {batch_data.shape}")
    encoded_batch = encode_batch(batch_data, n_qubits)
    print(f"Encoded batch shape: {encoded_batch.shape}")
    print(f"First encoded vector: {encoded_batch[0]}")
    print("-" * 50)
    
    # Example 3: Benchmarking
    print("Example 3: Performance Benchmarking")
    batch_sizes = [1, 10, 100, 1000]
    data_sizes = [32, 128, 512]
    
    results = []
    
    print(f"{'Batch Size':^10} | {'Data Size':^10} | {'NumPy (ms)':^12} | {'SIMD (ms)':^12} | {'Speedup':^10}")
    print("-" * 62)
    
    for batch_size in batch_sizes:
        for data_size in data_sizes:
            numpy_time, simd_time = benchmark(data_size, batch_size, n_qubits, n_runs=5)
            speedup = numpy_time / simd_time
            results.append((batch_size, data_size, numpy_time, simd_time, speedup))
            print(f"{batch_size:^10} | {data_size:^10} | {numpy_time:^12.3f} | {simd_time:^12.3f} | {speedup:^10.2f}x")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by batch size
    for batch_size in batch_sizes:
        batch_results = [r for r in results if r[0] == batch_size]
        data_sizes = [r[1] for r in batch_results]
        speedups = [r[4] for r in batch_results]
        ax.plot(data_sizes, speedups, 'o-', label=f'Batch Size = {batch_size}')
    
    ax.set_title('SIMD Angle Encoding Speedup vs. NumPy')
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Speedup Factor')
    ax.grid(True)
    ax.legend()
    
    try:
        plt.savefig('simd_speedup.png')
        print("\nSpeedup plot saved to 'simd_speedup.png'")
    except:
        print("\nCould not save plot")
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main() 