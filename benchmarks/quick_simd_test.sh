#!/bin/bash
echo "========================================"
echo "Quick SIMD Performance Comparison Test"
echo "========================================"
echo ""
echo "CPU Info:"
lscpu | grep -E "Architecture|Model name|Flags" | grep -E "Architecture|Model name"
echo ""
echo "SIMD Features:"
cat /proc/cpuinfo | grep "flags" | head -1 | tr ' ' '\n' | grep -E "sse|avx" | sort
echo ""
echo "========================================"
echo "Running Quick Performance Tests..."
echo "========================================"
echo ""

# Test different data sizes
for size in 16 64 256 1024; do
    echo "Testing data size: $size elements"
    python3 << PYEOF
import numpy as np
import sys
sys.path.insert(0, 'src')
import time

# Generate test data
data = np.random.rand(size).astype(np.float64)
n_qubits = size

# Import the module (will use best available SIMD)
try:
    from simd_angle_encoder import angle_encode_simd
    
    # Warmup
    for _ in range(100):
        _ = angle_encode_simd(data, n_qubits)
    
    # Measure
    times = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        result = angle_encode_simd(data, n_qubits)
        end = time.perf_counter_ns()
        times.append(end - start)
    
    avg_ns = np.mean(times)
    std_ns = np.std(times)
    throughput = (size * 1e9) / avg_ns
    
    print(f"  Optimized: {avg_ns:.2f} ± {std_ns:.2f} ns ({throughput:.0f} elems/sec)")
except Exception as e:
    print(f"  Error: {e}")
PYEOF
    echo ""
done

echo "========================================"
echo "Test Complete"
echo "========================================"
