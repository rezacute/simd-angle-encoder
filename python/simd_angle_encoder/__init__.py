"""
SIMD Angle Encoder - Fast quantum angle encoding using Rust SIMD optimizations

This module provides functions for angle encoding classical data for quantum computing,
optimized using SIMD instructions through Rust.
"""

import numpy as np
from typing import Union, Tuple

try:
    # Import the Rust module - note the name difference
    import simd_angle_encoder._simd_angle_encoder as _rust
    
    # Check that the functions exist
    _angle_encode_simd = _rust.angle_encode_simd
    _angle_encode_batch_simd = _rust.angle_encode_batch_simd
    _get_simd_info = _rust.get_simd_info
except (ImportError, AttributeError) as e:
    raise ImportError(
        f"Could not import SIMD angle encoder. Make sure the Rust extension is built. Error: {e}"
    )

def encode(data: Union[np.ndarray, list], n_qubits: int) -> np.ndarray:
    """
    Encode data using angle encoding with SIMD optimizations.
    
    Parameters
    ----------
    data : array-like
        The data to encode, must be convertible to a 1D numpy array
    n_qubits : int
        Number of qubits to use for encoding
        
    Returns
    -------
    np.ndarray
        Encoded data as angles in [0, 2π]
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)
    
    # Ensure 1D array
    if data.ndim > 1:
        data = data.flatten()
    
    # Ensure float64 dtype
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    
    return _angle_encode_simd(data, n_qubits)

def encode_batch(batch_data: Union[np.ndarray, list], n_qubits: int) -> np.ndarray:
    """
    Encode a batch of data using angle encoding with SIMD optimizations.
    
    Parameters
    ----------
    batch_data : array-like
        The batch of data to encode, must be convertible to a 2D numpy array
        with shape (batch_size, data_dim)
    n_qubits : int
        Number of qubits to use for encoding
        
    Returns
    -------
    np.ndarray
        Encoded data as angles in [0, 2π] with shape (batch_size, n_qubits)
    """
    # Convert to numpy array if needed
    if not isinstance(batch_data, np.ndarray):
        batch_data = np.array(batch_data, dtype=np.float64)
    
    # Ensure 2D array
    if batch_data.ndim == 1:
        batch_data = batch_data.reshape(1, -1)
    elif batch_data.ndim > 2:
        raise ValueError("batch_data must be a 1D or 2D array")
    
    # Ensure float64 dtype
    if batch_data.dtype != np.float64:
        batch_data = batch_data.astype(np.float64)
    
    return _angle_encode_batch_simd(batch_data, n_qubits)

def benchmark(data_size: int, batch_size: int, n_qubits: int, n_runs: int = 10) -> Tuple[float, float]:
    """
    Run a quick benchmark comparing numpy vs SIMD performance.
    
    Parameters
    ----------
    data_size : int
        Size of data to encode
    batch_size : int
        Batch size to use
    n_qubits : int
        Number of qubits
    n_runs : int
        Number of benchmark runs
        
    Returns
    -------
    Tuple[float, float]
        Execution time for numpy and SIMD implementations in milliseconds
    """
    import time
    
    # Generate random data
    batch_data = np.random.random((batch_size, data_size))
    two_pi = 2.0 * np.pi
    
    # Benchmark numpy implementation
    def numpy_encode_batch(batch_data, n_qubits):
        batch_size, data_dim = batch_data.shape
        result = np.zeros((batch_size, n_qubits))
        
        for b in range(batch_size):
            data = batch_data[b]
            for i in range(min(len(data), n_qubits)):
                result[b, i] = data[i] * two_pi
        
        return result
    
    # Warmup
    _ = numpy_encode_batch(batch_data, n_qubits)
    _ = encode_batch(batch_data, n_qubits)
    
    # Benchmark numpy
    start_time = time.time()
    for _ in range(n_runs):
        numpy_result = numpy_encode_batch(batch_data, n_qubits)
    numpy_time = (time.time() - start_time) / n_runs * 1000  # ms
    
    # Benchmark SIMD
    start_time = time.time()
    for _ in range(n_runs):
        simd_result = encode_batch(batch_data, n_qubits)
    simd_time = (time.time() - start_time) / n_runs * 1000  # ms
    
    # Verify results match
    assert np.allclose(numpy_result, simd_result, atol=1e-10)
    
    return numpy_time, simd_time

def simd_info() -> str:
    """
    Get information about SIMD support.
    
    Returns
    -------
    str
        Information about SIMD support
    """
    return _get_simd_info() 