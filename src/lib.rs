use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::f64::consts::PI;

/// Angle encode data in SIMD-optimized Rust
/// Maps data to angles in [0, 2π]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn angle_encode_simd<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    n_qubits: usize,
) -> Bound<'py, PyArray1<f64>> {
    // Try zero-copy access first, fallback to copy if non-contiguous
    let data_slice = if let Ok(slice) = data.as_slice() {
        // Zero-copy path (fast)
        slice
    } else {
        // Fallback to copy for non-contiguous arrays (rare case)
        &data.as_array().to_vec()
    };

    let result = simd_angle_encode(data_slice, n_qubits);
    result.into_pyarray(py)
}

/// Batch angle encode data in SIMD-optimized Rust
/// Maps batches of data to angles in [0, 2π]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn angle_encode_batch_simd<'py>(
    py: Python<'py>,
    batch_data: PyReadonlyArray2<f64>,
    n_qubits: usize,
) -> Bound<'py, PyArray2<f64>> {
    // Correctly extract dimensions from numpy array
    let shape = batch_data.shape();
    let batch_size = shape[0];

    let mut result = vec![0.0; batch_size * n_qubits];

    // Get array view (handles non-contiguous arrays)
    let data_array = batch_data.as_array();

    // Process each batch with optimized row access
    for b in 0..batch_size {
        // Try zero-copy row access, fallback to copy if non-contiguous
        let row = data_array.row(b);
        let batch_slice = if let Some(slice) = row.as_slice() {
            // Zero-copy path (fast)
            slice
        } else {
            // Fallback to copy for non-contiguous rows (rare case)
            &row.to_vec()
        };

        // Encode this batch
        let encoded = simd_angle_encode(batch_slice, n_qubits);

        // Copy to result
        for (i, &val) in encoded.iter().enumerate() {
            result[b * n_qubits + i] = val;
        }
    }

    // Reshape to (batch_size, n_qubits)
    let result_array = unsafe {
        ndarray::ArrayView2::from_shape_ptr((batch_size, n_qubits), result.as_ptr()).to_owned()
    };

    result_array.into_pyarray(py)
}

/// SIMD-optimized angle encoding with memory optimization
#[must_use]
pub fn simd_angle_encode(data: &[f64], n_qubits: usize) -> Vec<f64> {
    const SMALL_SIZE: usize = 32; // 256 bytes (fits in stack)

    // Fast path: Stack allocation for small data
    if n_qubits <= SMALL_SIZE {
        let mut result = [0.0f64; SMALL_SIZE];
        let two_pi = 2.0 * PI;

        // Process data in single pass (no inner loop!)
        let len = data.len().min(n_qubits);
        for i in 0..len {
            result[i] = data[i] * two_pi;
        }

        // Return only the needed portion
        return result[0..n_qubits].to_vec();
    }

    // Slow path: Heap allocation for large data
    let two_pi = 2.0 * PI;
    let mut result = Vec::with_capacity(n_qubits);

    // Simplified single-pass algorithm (removed inner loop)
    let len = data.len().min(n_qubits);
    for i in 0..len {
        result.push(data[i] * two_pi);
    }

    // Pad with zeros if needed
    result.resize(n_qubits, 0.0);

    result
}

/// Get information about SIMD support
#[pyfunction]
fn get_simd_info() -> String {
    // Detect SIMD support based on target architecture
    let simd_support = if cfg!(any(
        target_feature = "sse2",
        target_feature = "neon",
        target_feature = "simd128"
    )) {
        "Yes"
    } else {
        "Limited (compiler auto-vectorization only)"
    };

    // Get system information
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cpu_cores = num_cpus::get();

    format!("SIMD Support: {simd_support}\nSystem: {os}/{arch}\nCPU Cores: {cpu_cores}")
}

/// Python module for SIMD-optimized angle encoding
#[pymodule]
fn _simd_angle_encoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(angle_encode_simd, m)?)?;
    m.add_function(wrap_pyfunction!(angle_encode_batch_simd, m)?)?;
    m.add_function(wrap_pyfunction!(get_simd_info, m)?)?;
    Ok(())
}
