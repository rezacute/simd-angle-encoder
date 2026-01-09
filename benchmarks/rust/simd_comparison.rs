// Comprehensive SIMD comparison benchmark for Linux x86_64
//
// This benchmark rigorously compares all SIMD implementations:
// - Scalar (baseline)
// - AVX2 (256-bit, 4 doubles)
// - AVX-512 (512-bit, 8 doubles)
// - Auto-dispatch (runtime detection)

#![allow(clippy::cast_precision_loss, clippy::explicit_iter_loop)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simd_angle_encoder::simd_angle_encode;

/// Scalar implementation for baseline comparison
fn angle_encode_scalar(data: &[f64], n_qubits: usize) -> Vec<f64> {
    let two_pi = std::f64::consts::PI * 2.0;
    let mut result = Vec::with_capacity(n_qubits);

    let len = data.len().min(n_qubits);
    for &val in data.iter().take(len) {
        result.push(val * two_pi);
    }

    result.resize(n_qubits, 0.0);
    result
}

/// AVX2 implementation for comparison
#[cfg(target_arch = "x86_64")]
fn angle_encode_avx2_wrapper(data: &[f64], n_qubits: usize) -> Vec<f64> {
    if is_x86_feature_detected!("avx2") {
        unsafe { simd_angle_encoder::angle_encode_avx2(data, n_qubits) }
    } else {
        angle_encode_scalar(data, n_qubits)
    }
}

/// AVX-512 implementation for comparison
#[cfg(target_arch = "x86_64")]
fn angle_encode_avx512_wrapper(data: &[f64], n_qubits: usize) -> Vec<f64> {
    if is_x86_feature_detected!("avx512f") {
        unsafe { simd_angle_encoder::angle_encode_avx512(data, n_qubits) }
    } else {
        angle_encode_scalar(data, n_qubits)
    }
}

/// Generate random test data of specified size
fn generate_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            // Use deterministic pseudo-random values based on index
            ((i as f64) * 1.342_54_f64).fract()
        })
        .collect()
}

/// Benchmark all implementations for various data sizes
fn bench_simd_comparison_all_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_comparison_all_sizes");

    let sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    for size in &sizes {
        let data = generate_data(*size);

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &_size| {
            b.iter(|| black_box(angle_encode_scalar(black_box(&data), black_box(*size))));
        });

        // AVX2
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("avx2", size), size, |b, &_size| {
                b.iter(|| {
                    black_box(angle_encode_avx2_wrapper(
                        black_box(&data),
                        black_box(*size),
                    ))
                });
            });
        }

        // AVX-512
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            group.bench_with_input(BenchmarkId::new("avx512", size), size, |b, &_size| {
                b.iter(|| {
                    black_box(angle_encode_avx512_wrapper(
                        black_box(&data),
                        black_box(*size),
                    ))
                });
            });
        }

        // Auto-dispatch (optimized)
        group.bench_with_input(BenchmarkId::new("optimized", size), size, |b, &_size| {
            b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(*size))));
        });
    }

    group.finish();
}

/// Benchmark throughput (elements per second) for all implementations
fn bench_simd_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_throughput");

    let sizes = [64, 256, 1024, 4096, 16384];

    for size in &sizes {
        let data = generate_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &_size| {
            b.iter(|| black_box(angle_encode_scalar(black_box(&data), black_box(*size))));
        });

        // AVX2
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("avx2", size), size, |b, &_size| {
                b.iter(|| {
                    black_box(angle_encode_avx2_wrapper(
                        black_box(&data),
                        black_box(*size),
                    ))
                });
            });
        }

        // AVX-512
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            group.bench_with_input(BenchmarkId::new("avx512", size), size, |b, &_size| {
                b.iter(|| {
                    black_box(angle_encode_avx512_wrapper(
                        black_box(&data),
                        black_box(*size),
                    ))
                });
            });
        }

        // Auto-dispatch (optimized)
        group.bench_with_input(BenchmarkId::new("optimized", size), size, |b, &_size| {
            b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(*size))));
        });
    }

    group.finish();
}

/// Benchmark SIMD-friendly vs unfriendly sizes
fn bench_simd_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_alignment");

    // AVX2-friendly (multiples of 4)
    for size in [16, 32, 64, 128, 256].iter() {
        let data = generate_data(*size);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("avx2_aligned", size), size, |b, &_size| {
                b.iter(|| {
                    black_box(angle_encode_avx2_wrapper(
                        black_box(&data),
                        black_box(*size),
                    ))
                });
            });
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            group.bench_with_input(
                BenchmarkId::new("avx512_aligned", size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        black_box(angle_encode_avx512_wrapper(
                            black_box(&data),
                            black_box(*size),
                        ))
                    });
                },
            );
        }
    }

    // AVX2-unfriendly (not multiples of 4)
    for size in [17, 33, 65, 129, 257].iter() {
        let data = generate_data(*size);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            group.bench_with_input(
                BenchmarkId::new("avx2_unaligned", size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        black_box(angle_encode_avx2_wrapper(
                            black_box(&data),
                            black_box(*size),
                        ))
                    });
                },
            );
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            group.bench_with_input(
                BenchmarkId::new("avx512_unaligned", size),
                size,
                |b, &_size| {
                    b.iter(|| {
                        black_box(angle_encode_avx512_wrapper(
                            black_box(&data),
                            black_box(*size),
                        ))
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark head-to-head: Scalar vs SIMD for common sizes
fn bench_simd_head_to_head(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_head_to_head");

    let common_sizes = [8, 16, 32, 64, 128, 256];

    for size in &common_sizes {
        let data = generate_data(*size);

        // Compare scalar vs optimized directly
        group.bench_with_input(
            BenchmarkId::new("scalar_vs_optimized", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    black_box(angle_encode_scalar(black_box(&data), black_box(*size)));
                    black_box(simd_angle_encode(black_box(&data), black_box(*size)));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache behavior: small vs medium vs large
fn bench_simd_cache_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_cache_behavior");

    // Small (fits in L1 cache)
    let small = generate_data(16);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        group.bench_function("avx512_l1_friendly", |b| {
            b.iter(|| {
                black_box(angle_encode_avx512_wrapper(
                    black_box(&small),
                    black_box(16),
                ))
            });
        });
    }

    // Medium (fits in L2 cache)
    let medium = generate_data(512);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        group.bench_function("avx512_l2_friendly", |b| {
            b.iter(|| {
                black_box(angle_encode_avx512_wrapper(
                    black_box(&medium),
                    black_box(512),
                ))
            });
        });
    }

    // Large (exceeds L3 cache)
    let large = generate_data(16384);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        group.bench_function("avx512_cache_overflow", |b| {
            b.iter(|| {
                black_box(angle_encode_avx512_wrapper(
                    black_box(&large),
                    black_box(16384),
                ))
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = simd_benches;
    config = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(100);
    targets =
        bench_simd_comparison_all_sizes,
        bench_simd_throughput,
        bench_simd_alignment,
        bench_simd_head_to_head,
        bench_simd_cache_behavior
);

criterion_main!(simd_benches);
