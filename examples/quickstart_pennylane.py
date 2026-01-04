#!/usr/bin/env python3
"""
Quick Start Script for PennyLane SIMD Angle Encoder

This script verifies the installation and runs a simple example.
"""

import sys


def check_imports():
    """Check that all required packages can be imported."""
    print("Checking package availability...")

    # Check NumPy
    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False

    # Check PennyLane
    try:
        import pennylane as qml
        print(f"✓ PennyLane available (version {qml.__version__})")
    except ImportError:
        print("✗ PennyLane not found. Install with: pip install pennylane")
        return False

    # Check SIMD encoder
    try:
        from simd_angle_encoder import encode
        print("✓ SIMD Angle Encoder available")
    except ImportError:
        print("✗ SIMD Angle Encoder not found.")
        print("  Install with: pip install simd-angle-encoder")
        return False

    # Check PennyLane plugin
    try:
        from pennylane_simd_angle import AngleEncoding
        print("✓ PennyLane SIMD plugin available")
    except ImportError:
        print("✗ PennyLane SIMD plugin not found.")
        print("  The plugin files should be in python/pennylane_simd_angle/")
        return False

    return True


def run_simple_example():
    """Run a simple example to verify everything works."""
    print("\n" + "="*60)
    print("Running Simple Example")
    print("="*60)

    import numpy as np
    import pennylane as qml
    from pennylane_simd_angle import AngleEncoding

    # Create device
    dev = qml.device('default.qubit', wires=4)
    print(f"\n✓ Created device: {dev.name}")

    # Define circuit
    @qml.qnode(dev)
    def circuit(data):
        AngleEncoding(data, wires=range(4))
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        return qml.expval(qml.PauliZ(0))

    print("✓ Defined quantum circuit")

    # Execute circuit
    data = [0.1, 0.2, 0.3, 0.4]
    result = circuit(data)

    print(f"\n✓ Circuit executed successfully")
    print(f"  Input data: {data}")
    print(f"  Result: {result:.6f}")

    return True


def run_gradient_example():
    """Run a gradient computation example."""
    print("\n" + "="*60)
    print("Running Gradient Computation Example")
    print("="*60)

    from pennylane import numpy as pnp
    import pennylane as qml
    from pennylane_simd_angle import AngleEncoding

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(data):
        AngleEncoding(data, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    # Data with gradient tracking
    data = pnp.array([0.1, 0.2], requires_grad=True)

    # Compute gradient
    result = circuit(data)
    grad = qml.grad(circuit)(data)

    print(f"✓ Circuit output: {result:.6f}")
    print(f"✓ Gradient computed: {grad}")

    return True


def run_batch_example():
    """Run a batch processing example."""
    print("\n" + "="*60)
    print("Running Batch Processing Example")
    print("="*60)

    import numpy as np
    from pennylane_simd_angle.angle import angle_encoding_batch_function

    # Create batch
    batch = np.random.random((10, 4))

    # Encode batch
    angles = angle_encoding_batch_function(batch, n_qubits=4)

    print(f"✓ Batch shape: {batch.shape}")
    print(f"✓ Encoded shape: {angles.shape}")
    print(f"✓ Sample angles: {angles[0]}")

    return True


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("PennyLane SIMD Angle Encoder - Quick Start")
    print("="*60 + "\n")

    # Check imports
    if not check_imports():
        print("\n✗ Setup failed. Please install missing packages.")
        sys.exit(1)

    print("\n✓ All packages available!")

    # Run examples
    try:
        run_simple_example()
        run_gradient_example()
        run_batch_example()

        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60 + "\n")

        print("Next steps:")
        print("  1. See examples/pennylane_examples.py for more examples")
        print("  2. Read docs/pennylane-integration.md for full documentation")
        print("  3. Run benchmarks/pennylane_benchmark.py for performance metrics")
        print()

    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
