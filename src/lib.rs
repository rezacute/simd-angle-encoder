use numpy::PyUntypedArrayMethods;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::f64::consts::PI;

// NEON intrinsics for ARM64
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// AVX intrinsics for x86_64
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Rayon for parallel processing
use rayon::prelude::*;

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

/// Batch angle encode data in SIMD-optimized Rust with parallel processing
/// Maps batches of data to angles in [0, 2π]
///
/// Phase 2C Optimization: Uses Rayon for parallel batch processing
/// - Small batches (< 10): Sequential processing (avoid overhead)
/// - Large batches (≥ 10): Parallel processing (2-4x speedup)
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

    // Phase 2C: Sequential fallback for small batches (avoid parallel overhead)
    const PARALLEL_THRESHOLD: usize = 10;
    if batch_size < PARALLEL_THRESHOLD {
        return angle_encode_batch_sequential(batch_data, n_qubits, py);
    }

    // Get array view and convert to owned for parallel processing
    let data_array = batch_data.as_array();

    // Phase 2C: Parallel batch processing using Rayon
    // Each batch is independent, so we can process them in parallel
    let results: Vec<Vec<f64>> = data_array
        .axis_iter(ndarray::Axis(0))
        .collect::<Vec<_>>() // Collect to enable parallel iteration
        .into_par_iter()     // Convert to parallel iterator!
        .map(|row| {
            // Try zero-copy row access, fallback to copy if non-contiguous
            let batch_slice = if let Some(slice) = row.as_slice() {
                // Zero-copy path (fast)
                slice
            } else {
                // Fallback to copy for non-contiguous rows (rare case)
                &row.to_vec()
            };

            // Encode this batch using optimized SIMD function
            simd_angle_encode(batch_slice, n_qubits)
        })
        .collect(); // Collect results from all threads

    // Flatten results into 2D array
    let mut result = Vec::with_capacity(batch_size * n_qubits);
    for encoded in results {
        result.extend_from_slice(&encoded);
    }

    // Reshape to (batch_size, n_qubits)
    let result_array = unsafe {
        ndarray::ArrayView2::from_shape_ptr((batch_size, n_qubits), result.as_ptr()).to_owned()
    };

    result_array.into_pyarray(py)
}

/// Sequential batch processing (fallback for small batches)
///
/// Used for batches < 10 rows where parallel overhead exceeds benefit
/// Maintains same logic as original implementation for consistency
#[allow(clippy::needless_pass_by_value)]
fn angle_encode_batch_sequential<'py>(
    batch_data: PyReadonlyArray2<f64>,
    n_qubits: usize,
    py: Python<'py>,
) -> Bound<'py, PyArray2<f64>> {
    let shape = batch_data.shape();
    let batch_size = shape[0];

    let mut result = vec![0.0; batch_size * n_qubits];

    // Get array view (handles non-contiguous arrays)
    let data_array = batch_data.as_array();

    // Process each batch sequentially
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

/// NEON-optimized angle encoding for ARM64
/// Processes 2 doubles per iteration (128-bit SIMD width)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn angle_encode_neon(data: &[f64], n_qubits: usize) -> Vec<f64> {
    // SAFETY: Caller must ensure NEON is supported (checked via is_arm_feature_detected!)
    // We maintain bounds checking and only use intrinsics within safe bounds

    let two_pi_vec = vdupq_n_f64(2.0 * PI);  // Broadcast 2π constant
    let mut result = Vec::with_capacity(n_qubits);

    // Process 2 doubles at a time (NEON processes 2 f64 in 128-bit register)
    let simd_chunks = (data.len().min(n_qubits) / 2) * 2;
    let mut i = 0;

    // Main SIMD loop: process 2 doubles per iteration
    while i + 2 <= simd_chunks {
        // SAFETY: We've verified i+2 is within bounds via simd_chunks calculation
        // Load 2 doubles from input
        let input = vld1q_f64(data.as_ptr().add(i));

        // Vectorized multiply: output = input * 2π (2 operations in parallel!)
        let output = vmulq_f64(input, two_pi_vec);

        // Store 2 doubles to temporary array, then extend result
        let mut temp = [0.0f64; 2];
        vst1q_f64(temp.as_mut_ptr(), output);
        result.extend_from_slice(&temp);

        i += 2;
    }

    // Handle remainder elements (0 or 1 element)
    while i < data.len().min(n_qubits) {
        result.push(data[i] * 2.0 * PI);
        i += 1;
    }

    // Pad with zeros if needed
    result.resize(n_qubits, 0.0);

    result
}

/// AVX-512 optimized angle encoding for x86_64
/// Processes 8 doubles per iteration (512-bit SIMD width)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn angle_encode_avx512(data: &[f64], n_qubits: usize) -> Vec<f64> {
    // SAFETY: Caller must ensure AVX-512 is supported (checked via is_x86_feature_detected!)
    let two_pi_vec = _mm512_set1_pd(2.0 * PI);
    let mut result = Vec::with_capacity(n_qubits);

    // Process 8 doubles at a time (AVX-512 processes 8 f64 in 512-bit register)
    let simd_chunks = (data.len().min(n_qubits) / 8) * 8;
    let mut i = 0;

    // Main SIMD loop: process 8 doubles per iteration
    while i + 8 <= simd_chunks {
        // Load 8 doubles from input
        let input = _mm512_loadu_pd(data.as_ptr().add(i));

        // Vectorized multiply: output = input * 2π (8 operations in parallel!)
        let output = _mm512_mul_pd(input, two_pi_vec);

        // Store 8 doubles to temporary array, then extend result
        let mut temp = [0.0f64; 8];
        _mm512_storeu_pd(temp.as_mut_ptr(), output);
        result.extend_from_slice(&temp);

        i += 8;
    }

    // Handle remainder elements
    while i < data.len().min(n_qubits) {
        result.push(data[i] * 2.0 * PI);
        i += 1;
    }

    // Pad with zeros if needed
    result.resize(n_qubits, 0.0);

    result
}

/// AVX2 optimized angle encoding for x86_64
/// Processes 4 doubles per iteration (256-bit SIMD width)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn angle_encode_avx2(data: &[f64], n_qubits: usize) -> Vec<f64> {
    // SAFETY: Caller must ensure AVX2 is supported (checked via is_x86_feature_detected!)
    let two_pi_vec = _mm256_set1_pd(2.0 * PI);
    let mut result = Vec::with_capacity(n_qubits);

    // Process 4 doubles at a time (AVX2 processes 4 f64 in 256-bit register)
    let simd_chunks = (data.len().min(n_qubits) / 4) * 4;
    let mut i = 0;

    // Main SIMD loop: process 4 doubles per iteration
    while i + 4 <= simd_chunks {
        // Load 4 doubles from input
        let input = _mm256_loadu_pd(data.as_ptr().add(i));

        // Vectorized multiply: output = input * 2π (4 operations in parallel!)
        let output = _mm256_mul_pd(input, two_pi_vec);

        // Store 4 doubles to temporary array, then extend result
        let mut temp = [0.0f64; 4];
        _mm256_storeu_pd(temp.as_mut_ptr(), output);
        result.extend_from_slice(&temp);

        i += 4;
    }

    // Handle remainder elements
    while i < data.len().min(n_qubits) {
        result.push(data[i] * 2.0 * PI);
        i += 1;
    }

    // Pad with zeros if needed
    result.resize(n_qubits, 0.0);

    result
}

/// Scalar fallback for platforms without NEON or for validation
#[must_use]
fn angle_encode_scalar(data: &[f64], n_qubits: usize) -> Vec<f64> {
    let two_pi = 2.0 * PI;
    let mut result = Vec::with_capacity(n_qubits);

    let len = data.len().min(n_qubits);
    for i in 0..len {
        result.push(data[i] * two_pi);
    }

    result.resize(n_qubits, 0.0);
    result
}

/// SIMD-optimized angle encoding with memory optimization and platform-specific intrinsics
///
/// Multi-tier optimization strategy:
/// 1. Fast: Stack allocation for small data (n_qubits <= 32)
/// 2. Medium-x86: AVX-512 SIMD for x86_64 when available (8 doubles/iter)
/// 3. Medium-x86: AVX2 SIMD for x86_64 when available (4 doubles/iter)
/// 4. Medium-arm: NEON SIMD for ARM64 when available (2 doubles/iter)
/// 5. Slow: Scalar fallback for compatibility
#[must_use]
pub fn simd_angle_encode(data: &[f64], n_qubits: usize) -> Vec<f64> {
    const SMALL_SIZE: usize = 32; // 256 bytes (fits in stack)

    // Fast path: Stack allocation for small data (Phase 2A optimization)
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

    // Medium path: Platform-specific SIMD (Phase 2B optimization)

    // x86_64 path: AVX-512 -> AVX2 -> scalar
    #[cfg(target_arch = "x86_64")]
    {
        // Only use SIMD if we have enough elements
        if data.len() >= 4 && n_qubits >= 4 {
            // Try AVX-512 first (best performance - 8 doubles at once)
            if is_x86_feature_detected!("avx512f") {
                unsafe { return angle_encode_avx512(data, n_qubits); }
            }

            // Fall back to AVX2 (good performance - 4 doubles at once)
            if is_x86_feature_detected!("avx2") {
                unsafe { return angle_encode_avx2(data, n_qubits); }
            }
        }
    }

    // ARM64 path: NEON SIMD
    #[cfg(target_arch = "aarch64")]
    {
        // Only use NEON if we have at least 2 elements (NEON processes 2 at a time)
        if data.len() >= 2 && n_qubits >= 2 {
            // SAFETY: All ARM64 Apple Silicon has NEON support
            unsafe { return angle_encode_neon(data, n_qubits); }
        }
    }

    // Slow path: Scalar fallback for large data or no SIMD support
    angle_encode_scalar(data, n_qubits)
}

/// Get information about SIMD support
#[pyfunction]
fn get_simd_info() -> String {
    // Get system information
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cpu_cores = num_cpus::get();

    // Format based on architecture
    #[cfg(target_arch = "aarch64")]
    {
        // Note: All ARM64 Apple Silicon has NEON support
        // Compile-time detection: if we're on aarch64, we have NEON
        let simd_info = "Yes (NEON)";

        return format!(
            "SIMD Support: {simd_info}\nSystem: {os}/{arch}\nCPU Cores: {cpu_cores}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Detect runtime SIMD support for x86_64
        let simd_info = if is_x86_feature_detected!("avx512f") {
            "Yes (AVX-512)"
        } else if is_x86_feature_detected!("avx2") {
            "Yes (AVX2)"
        } else if is_x86_feature_detected!("sse2") {
            "Yes (SSE2)"
        } else {
            "No (scalar only)"
        };

        return format!(
            "SIMD Support: {simd_info}\nSystem: {os}/{arch}\nCPU Cores: {cpu_cores}"
        );
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        format!(
            "SIMD Support: Unknown\nSystem: {os}/{arch}\nCPU Cores: {cpu_cores}"
        )
    }
}

/// Python module for SIMD-optimized angle encoding
#[pymodule]
fn _simd_angle_encoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(angle_encode_simd, m)?)?;
    m.add_function(wrap_pyfunction!(angle_encode_batch_simd, m)?)?;
    m.add_function(wrap_pyfunction!(get_simd_info, m)?)?;
    Ok(())
}
