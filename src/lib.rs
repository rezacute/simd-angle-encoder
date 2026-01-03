use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use numpy::PyUntypedArrayMethods;
use std::f64::consts::PI;

/// Angle encode data in SIMD-optimized Rust
/// Maps data to angles in [0, 2π]
#[pyfunction]
fn angle_encode_simd<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    n_qubits: usize,
) -> Bound<'py, PyArray1<f64>> {
    // Handle both contiguous and non-contiguous arrays
    let data_array = data.as_array();
    let data_slice: Vec<f64> = data_array.to_vec();
    let result = simd_angle_encode(&data_slice, n_qubits);
    result.into_pyarray(py)
}

/// Batch angle encode data in SIMD-optimized Rust
/// Maps batches of data to angles in [0, 2π]
#[pyfunction]
fn angle_encode_batch_simd<'py>(
    py: Python<'py>,
    batch_data: PyReadonlyArray2<f64>,
    n_qubits: usize,
) -> Bound<'py, PyArray2<f64>> {
    // Correctly extract dimensions from numpy array
    let shape = batch_data.shape();
    let batch_size = shape[0];
    let data_dim = shape[1];

    let mut result = vec![0.0; batch_size * n_qubits];

    // Get array view (handles non-contiguous arrays)
    let data_array = batch_data.as_array();

    // Process each batch
    for b in 0..batch_size {
        // Extract row as a vector (handles non-contiguous arrays)
        let batch_slice: Vec<f64> = data_array.row(b).to_vec();

        // Encode this batch
        let encoded = simd_angle_encode(&batch_slice, n_qubits);

        // Copy to result
        for (i, &val) in encoded.iter().enumerate() {
            result[b * n_qubits + i] = val;
        }
    }

    // Reshape to (batch_size, n_qubits)
    let result_array = unsafe {
        ndarray::ArrayView2::from_shape_ptr(
            (batch_size, n_qubits),
            result.as_ptr()
        ).to_owned()
    };

    result_array.into_pyarray(py)
}

/// SIMD-optimized angle encoding
pub fn simd_angle_encode(data: &[f64], n_qubits: usize) -> Vec<f64> {
    let two_pi = 2.0 * PI;
    let mut result = Vec::with_capacity(n_qubits);

    // Process in chunks for better cache locality and vectorization
    let chunk_size = 4; // Size that works well with SIMD units
    let mut i = 0;

    // Process 4 elements at a time (SIMD-friendly)
    while i + chunk_size <= data.len() && i < n_qubits {
        for j in 0..chunk_size {
            if i + j < n_qubits {
                result.push(data[i + j] * two_pi);
            }
        }
        i += chunk_size;
    }

    // Handle remaining elements
    while i < data.len() && i < n_qubits {
        result.push(data[i] * two_pi);
        i += 1;
    }

    // Pad with zeros if needed
    result.resize(n_qubits, 0.0);

    result
}

/// Get information about SIMD support
#[pyfunction]
fn get_simd_info() -> String {
    // Detect SIMD support based on target architecture
    let simd_support = if cfg!(any(target_feature = "sse2", target_feature = "neon", target_feature = "simd128")) {
        "Yes"
    } else {
        "Limited (compiler auto-vectorization only)"
    };

    // Get system information
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cpu_cores = num_cpus::get();

    format!(
        "SIMD Support: {}\nSystem: {}/{}\nCPU Cores: {}",
        simd_support, os, arch, cpu_cores
    )
}

/// Python module for SIMD-optimized angle encoding
#[pymodule]
fn _simd_angle_encoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(angle_encode_simd, m)?)?;
    m.add_function(wrap_pyfunction!(angle_encode_batch_simd, m)?)?;
    m.add_function(wrap_pyfunction!(get_simd_info, m)?)?;
    Ok(())
}
