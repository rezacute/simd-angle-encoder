"""
Benchmark comparing PennyLane SIMD plugin vs standard PennyLane encoding.

This script measures the performance improvement of using the SIMD-accelerated
angle encoding compared to PennyLane's built-in NumPy-based encoding.
"""

import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

try:
    from pennylane_simd_angle import AngleEncoding, angle_encoding_function
    SIMD_AVAILABLE = True
except ImportError:
    print("Warning: PennyLane SIMD plugin not available, only benchmarking standard encoding")
    SIMD_AVAILABLE = False


def standard_angle_encoding(data):
    """Standard NumPy-based angle encoding."""
    return data * 2 * np.pi


def benchmark_encoding_function(n_qubits=10, n_runs=100):
    """Benchmark the encoding function itself."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Encoding Function (n_qubits={n_qubits}, runs={n_runs})")
    print(f"{'='*60}")

    data = np.random.random(n_qubits)

    # Standard encoding
    start = time.time()
    for _ in range(n_runs):
        angles_standard = standard_angle_encoding(data)
    time_standard = (time.time() - start) / n_runs * 1000  # ms

    print(f"Standard NumPy: {time_standard:.4f} ms per call")

    # SIMD encoding
    if SIMD_AVAILABLE:
        start = time.time()
        for _ in range(n_runs):
            angles_simd = angle_encoding_function(data, n_qubits=n_qubits)
        time_simd = (time.time() - start) / n_runs * 1000  # ms

        speedup = time_standard / time_simd
        print(f"SIMD Encoding:  {time_simd:.4f} ms per call")
        print(f"Speedup:        {speedup:.2f}x")

        # Verify correctness
        assert np.allclose(angles_standard, angles_simd, atol=1e-10)
        print("✓ Results match")
    else:
        print("SIMD encoding not available")


def benchmark_qnode_execution(n_qubits=10, n_runs=100):
    """Benchmark QNode execution with encoding."""
    print(f"\n{'='*60}")
    print(f"Benchmark: QNode Execution (n_qubits={n_qubits}, runs={n_runs})")
    print(f"{'='*60}")

    dev = qml.device('default.qubit', wires=n_qubits)

    # Standard circuit
    @qml.qnode(dev)
    def circuit_standard(data):
        angles = standard_angle_encoding(data)
        for i in range(n_qubits):
            qml.RY(angles[i], wires=i)
        return qml.expval(qml.PauliZ(0))

    # SIMD circuit
    if SIMD_AVAILABLE:
        @qml.qnode(dev)
        def circuit_simd(data):
            AngleEncoding(data, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

    data = np.random.random(n_qubits)

    # Benchmark standard
    start = time.time()
    for _ in range(n_runs):
        result_standard = circuit_standard(data)
    time_standard = (time.time() - start) / n_runs * 1000  # ms

    print(f"Standard Encoding: {time_standard:.4f} ms per QNode execution")

    # Benchmark SIMD
    if SIMD_AVAILABLE:
        start = time.time()
        for _ in range(n_runs):
            result_simd = circuit_simd(data)
        time_simd = (time.time() - start) / n_runs * 1000  # ms

        speedup = time_standard / time_simd
        print(f"SIMD Encoding:    {time_simd:.4f} ms per QNode execution")
        print(f"Speedup:          {speedup:.2f}x")

        # Verify correctness
        assert np.allclose(result_standard, result_simd, atol=1e-10)
        print("✓ Results match")
    else:
        print("SIMD encoding not available")


def benchmark_gradient_computation(n_qubits=4, n_runs=50):
    """Benchmark gradient computation."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Gradient Computation (n_qubits={n_qubits}, runs={n_runs})")
    print(f"{'='*60}")

    dev = qml.device('default.qubit', wires=n_qubits)

    # Standard circuit
    @qml.qnode(dev, diff_method="backprop")
    def circuit_standard(data):
        angles = data * 2 * np.pi
        for i in range(n_qubits):
            qml.RY(angles[i], wires=i)
        return qml.expval(qml.PauliZ(0))

    # SIMD circuit
    if SIMD_AVAILABLE:
        @qml.qnode(dev, diff_method="backprop")
        def circuit_simd(data):
            AngleEncoding(data, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

    data = pnp.array(np.random.random(n_qubits), requires_grad=True)

    # Benchmark standard
    start = time.time()
    for _ in range(n_runs):
        grad_standard = qml.grad(circuit_standard)(data)
    time_standard = (time.time() - start) / n_runs * 1000  # ms

    print(f"Standard Encoding: {time_standard:.4f} ms per gradient")

    # Benchmark SIMD
    if SIMD_AVAILABLE:
        start = time.time()
        for _ in range(n_runs):
            grad_simd = qml.grad(circuit_simd)(data)
        time_simd = (time.time() - start) / n_runs * 1000  # ms

        speedup = time_standard / time_simd
        print(f"SIMD Encoding:    {time_simd:.4f} ms per gradient")
        print(f"Speedup:          {speedup:.2f}x")

        # Verify correctness
        assert np.allclose(grad_standard, grad_simd, atol=1e-10)
        print("✓ Results match")
    else:
        print("SIMD encoding not available")


def benchmark_batch_encoding(batch_size=100, n_qubits=10, n_runs=50):
    """Benchmark batch encoding."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Batch Encoding (batch={batch_size}, n_qubits={n_qubits}, runs={n_runs})")
    print(f"{'='*60}")

    batch_data = np.random.random((batch_size, n_qubits))

    # Standard encoding
    def standard_encode_batch(batch):
        return batch * 2 * np.pi

    start = time.time()
    for _ in range(n_runs):
        angles_standard = standard_encode_batch(batch_data)
    time_standard = (time.time() - start) / n_runs * 1000  # ms

    print(f"Standard NumPy: {time_standard:.4f} ms per batch")

    # SIMD encoding
    if SIMD_AVAILABLE:
        from pennylane_simd_angle.angle import angle_encoding_batch_function

        start = time.time()
        for _ in range(n_runs):
            angles_simd = angle_encoding_batch_function(batch_data, n_qubits=n_qubits)
        time_simd = (time.time() - start) / n_runs * 1000  # ms

        speedup = time_standard / time_simd
        print(f"SIMD Encoding:  {time_simd:.4f} ms per batch")
        print(f"Speedup:        {speedup:.2f}x")

        # Verify correctness
        assert np.allclose(angles_standard, angles_simd, atol=1e-10)
        print("✓ Results match")
    else:
        print("SIMD encoding not available")


def benchmark_variational_circuit(n_qubits=8, n_runs=100):
    """Benchmark a realistic variational quantum circuit."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Variational Circuit (n_qubits={n_qubits}, runs={n_runs})")
    print(f"{'='*60}")

    dev = qml.device('default.qubit', wires=n_qubits)

    # Variational parameters
    params = pnp.array(np.random.random(n_qubits) * 0.1, requires_grad=True)

    # Standard circuit with encoding
    @qml.qnode(dev, diff_method="backprop")
    def circuit_standard(data, params):
        # Encoding layer
        angles = data * 2 * np.pi
        for i in range(n_qubits):
            qml.RY(angles[i], wires=i)

        # Variational layer
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    # SIMD circuit with encoding
    if SIMD_AVAILABLE:
        @qml.qnode(dev, diff_method="backprop")
        def circuit_simd(data, params):
            # Encoding layer
            AngleEncoding(data, wires=range(n_qubits))

            # Variational layer
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            return qml.expval(qml.PauliZ(0))

    data = pnp.array(np.random.random(n_qubits), requires_grad=True)

    # Benchmark standard
    start = time.time()
    for _ in range(n_runs):
        result_standard = circuit_standard(data, params)
    time_standard = (time.time() - start) / n_runs * 1000  # ms

    print(f"Standard Encoding: {time_standard:.4f} ms per QNode")

    # Benchmark SIMD
    if SIMD_AVAILABLE:
        start = time.time()
        for _ in range(n_runs):
            result_simd = circuit_simd(data, params)
        time_simd = (time.time() - start) / n_runs * 1000  # ms

        speedup = time_standard / time_simd
        print(f"SIMD Encoding:    {time_simd:.4f} ms per QNode")
        print(f"Speedup:          {speedup:.2f}x")

        # Verify correctness
        assert np.allclose(result_standard, result_simd, atol=1e-10)
        print("✓ Results match")
    else:
        print("SIMD encoding not available")


def run_all_benchmarks():
    """Run all benchmarks with different configurations."""
    print("\n" + "="*60)
    print("PennyLane SIMD Angle Encoder - Performance Benchmarks")
    print("="*60)

    if not SIMD_AVAILABLE:
        print("\n⚠ Warning: SIMD plugin not available. Install with:")
        print("   pip install simd-angle-encoder pennylane-simd-angle-encoder")

    # Test different scales
    configs = [
        {"n_qubits": 4, "n_runs": 200},
        {"n_qubits": 8, "n_runs": 150},
        {"n_qubits": 16, "n_runs": 100},
        {"n_qubits": 32, "n_runs": 50},
    ]

    for config in configs:
        benchmark_encoding_function(**config)
        benchmark_qnode_execution(**config)

    # Gradient benchmarks
    benchmark_gradient_computation(n_qubits=4, n_runs=50)
    benchmark_gradient_computation(n_qubits=8, n_runs=30)

    # Batch benchmarks
    batch_configs = [
        {"batch_size": 10, "n_qubits": 4, "n_runs": 100},
        {"batch_size": 50, "n_qubits": 8, "n_runs": 50},
        {"batch_size": 100, "n_qubits": 16, "n_runs": 30},
    ]

    for config in batch_configs:
        benchmark_batch_encoding(**config)

    # Variational circuit benchmarks
    benchmark_variational_circuit(n_qubits=4, n_runs=100)
    benchmark_variational_circuit(n_qubits=8, n_runs=50)

    print("\n" + "="*60)
    print("Benchmarking Complete!")
    print("="*60)


if __name__ == "__main__":
    run_all_benchmarks()
