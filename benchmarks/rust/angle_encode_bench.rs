// Criterion.rs benchmarks for SIMD angle encoder
//
// Low-level Rust microbenchmarks to measure the performance of
// the core encoding functions without Python overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simd_angle_encoder::simd_angle_encode;

/// Generate random test data of specified size
fn generate_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            // Use deterministic pseudo-random values based on index
            let x = ((i as f64) * 1.3_4254_f64).fract();
            x
        })
        .collect()
}

/// Benchmark single encode function with various data sizes
fn bench_encode_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_single");

    for size in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096].iter() {
        let data = generate_data(*size);
        let n_qubits = *size;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(n_qubits))));
        });
    }

    group.finish();
}

/// Benchmark encode with fixed data size but varying qubit counts
fn bench_encode_qubits(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_qubits");

    let data = generate_data(256); // Fixed data size

    for n_qubits in [4, 8, 16, 32, 64, 128, 256].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_qubits),
            n_qubits,
            |b, &n_qubits| {
                b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(n_qubits))));
            },
        );
    }

    group.finish();
}

/// Benchmark encode when data is smaller than n_qubits
fn bench_encode_data_smaller(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_data_smaller");

    // Data size = 16, n_qubits = 64
    let data = generate_data(16);
    let n_qubits = 64;

    group.bench_function("data_16_qubits_64", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(n_qubits))));
    });

    group.finish();
}

/// Benchmark encode when data is larger than n_qubits
fn bench_encode_data_larger(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_data_larger");

    // Data size = 256, n_qubits = 16
    let data = generate_data(256);
    let n_qubits = 16;

    group.bench_function("data_256_qubits_16", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(n_qubits))));
    });

    group.finish();
}

/// Benchmark batch encoding simulation (repeated encode calls)
fn bench_encode_batch_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_batch_simulation");

    // Simulate different batch sizes
    for batch_size in [10, 50, 100, 500, 1000].iter() {
        let data_dim = 64;
        let n_qubits = 64;

        // Pre-generate all batches
        let batches: Vec<Vec<f64>> = (0..*batch_size).map(|_| generate_data(data_dim)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &_batch_size| {
                b.iter(|| {
                    // Encode all batches
                    for batch_data in &batches {
                        black_box(simd_angle_encode(
                            black_box(batch_data),
                            black_box(n_qubits),
                        ));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark edge cases
fn bench_encode_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_edge_cases");

    // All zeros
    let zeros: Vec<f64> = vec![0.0; 64];
    group.bench_function("zeros_64", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&zeros), black_box(64))));
    });

    // All ones
    let ones: Vec<f64> = vec![1.0; 64];
    group.bench_function("ones_64", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&ones), black_box(64))));
    });

    // Mixed values including edge values
    let mixed: Vec<f64> = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.5];
    group.bench_function("mixed_8", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&mixed), black_box(8))));
    });

    // Single value
    let single: Vec<f64> = vec![0.5];
    group.bench_function("single_1", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&single), black_box(1))));
    });

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_encode_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_memory_patterns");

    // Small allocations (fits in cache)
    let small_data = generate_data(16);
    group.bench_function("cache_friendly_small", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&small_data), black_box(16))));
    });

    // Medium allocations
    let medium_data = generate_data(256);
    group.bench_function("cache_friendly_medium", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&medium_data), black_box(256))));
    });

    // Large allocations (may exceed cache)
    let large_data = generate_data(4096);
    group.bench_function("cache_unfriendly_large", |b| {
        b.iter(|| black_box(simd_angle_encode(black_box(&large_data), black_box(4096))));
    });

    group.finish();
}

/// Benchmark SIMD chunked processing
fn bench_encode_simd_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_simd_chunks");

    // Sizes that are multiples of 4 (SIMD-friendly)
    for size in [16, 32, 64, 128, 256, 512].iter() {
        let data = generate_data(*size);

        group.bench_with_input(
            BenchmarkId::new("multiple_of_4", size),
            size,
            |b, &_size| {
                b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(*size))));
            },
        );
    }

    // Sizes that are NOT multiples of 4 (requires remainder handling)
    for size in [17, 33, 65, 129, 257, 513].iter() {
        let data = generate_data(*size);

        group.bench_with_input(
            BenchmarkId::new("not_multiple_of_4", size),
            size,
            |b, &_size| {
                b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(*size))));
            },
        );
    }

    group.finish();
}

/// Benchmark throughput (elements per second)
fn bench_encode_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_throughput");

    // Measure throughput for different sizes
    for size in [64, 256, 1024, 4096, 16384].iter() {
        let data = generate_data(*size);

        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| black_box(simd_angle_encode(black_box(&data), black_box(*size))));
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(100);
    targets =
        bench_encode_single,
        bench_encode_qubits,
        bench_encode_data_smaller,
        bench_encode_data_larger,
        bench_encode_batch_simulation,
        bench_encode_edge_cases,
        bench_encode_memory_patterns,
        bench_encode_simd_chunks,
        bench_encode_throughput
);

criterion_main!(benches);
