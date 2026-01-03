# SIMD Angle Encoder Benchmark Suite

Comprehensive benchmark suite for measuring and validating the performance of the SIMD-optimized angle encoder implementation.

## Overview

This benchmark suite provides:
- **Micro-benchmarks**: Individual function performance measurements
- **Macro-benchmarks**: Real-world workload simulations
- **Comparative benchmarks**: Performance vs NumPy baseline
- **Scalability benchmarks**: Performance across different data sizes and batch sizes
- **Regression detection**: Automated performance regression detection in CI/CD

## Directory Structure

```
benchmarks/
├── README.md                           # This file
├── pytest.ini                          # Pytest benchmark configuration
├── conftest.py                         # Benchmark fixtures and configuration
├── python/                             # Python benchmarks (pytest-benchmark)
│   ├── test_bench_encode.py           # Single encoding benchmarks
│   ├── test_bench_encode_batch.py     # Batch encoding benchmarks
│   ├── test_bench_comparative.py      # Comparative benchmarks vs NumPy
│   └── test_bench_scalability.py      # Scalability benchmarks
├── rust/                               # Rust benchmarks (criterion.rs)
│   └── angle_encode_bench.rs          # Rust microbenchmarks
└── reports/                            # Generated benchmark reports
    ├── .gitkeep
    └── (generated reports)
```

## Installation

### Additional Dependencies

```bash
# Install pytest-benchmark
pip install pytest-benchmark

# Install plotting dependencies (optional, for generating plots)
pip install matplotlib plotly
```

### Rust Benchmarks

Criterion.rs is included as a dev dependency in `Cargo.toml` and will be automatically installed.

## Running Benchmarks

### Run All Python Benchmarks

```bash
# Run all benchmarks with default settings
pytest benchmarks/ -v

# Run with more iterations for better statistical accuracy
pytest benchmarks/ --benchmark-min-rounds=50 -v

# Run only specific benchmark categories
pytest benchmarks/python/test_bench_encode.py -v
pytest benchmarks/python/test_bench_encode_batch.py -v
pytest benchmarks/python/test_bench_comparative.py -v
pytest benchmarks/python/test_bench_scalability.py -v
```

### Generate Benchmark Reports

```bash
# Generate HTML report
pytest benchmarks/ --benchmark-only --benchmark-autosave \
  --benchmark-json=reports/benchmark-results.json

# Generate comparison with previous run
pytest benchmarks/ --benchmark-compare \
  --benchmark-compare-fail=mean:10% \
  --benchmark-autosave

# Generate histograms
pytest benchmarks/ --benchmark-only --benchmark-histogram
```

### Run Rust Benchmarks

```bash
# Run criterion benchmarks
cargo bench --bench angle_encode_bench

# Generate specific benchmark report
cargo bench --bench angle_encode_bench -- --save-baseline main

# Compare against baseline
cargo bench --bench angle_encode_bench -- --baseline main
```

## Benchmark Categories

### 1. Single Encoding Benchmarks (`test_bench_encode.py`)

Measures performance of `encode()` for various data sizes and qubit counts.

**Test scenarios:**
- Data sizes: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
- Qubit counts: 4, 8, 16, 32
- Edge cases: single value, small arrays, large arrays

### 2. Batch Encoding Benchmarks (`test_bench_encode_batch.py`)

Measures performance of `encode_batch()` across different batch sizes and dimensions.

**Test scenarios:**
- Batch sizes: 1, 10, 50, 100, 500, 1000, 5000, 10000
- Data dimensions: 8, 16, 32, 64, 128, 256
- Qubit counts: 4, 8, 16, 32

### 3. Comparative Benchmarks (`test_bench_comparative.py`)

Compares SIMD implementation against NumPy baseline.

**Test scenarios:**
- Small data (64 elements)
- Medium data (1024 elements)
- Large data (8192 elements)
- Various batch sizes (1 to 10000)
- Speedup factor calculation

### 4. Scalability Benchmarks (`test_bench_scalability.py`)

Analyzes performance scaling characteristics.

**Test scenarios:**
- Linear scaling tests (data size vs time)
- Batch scaling tests (batch size vs time)
- Dimension scaling tests (qubit count vs time)
- Memory efficiency tests

### 5. Rust Microbenchmarks (`angle_encode_bench.rs`)

Low-level Rust function benchmarks using criterion.rs.

**Test scenarios:**
- Core `simd_angle_encode()` function
- Various input sizes
- Compiler optimization validation

## Understanding Results

### pytest-benchmark Output

```
-------------------------------------------------------------------------------------------
Name (time in ms)                          Min       Max      Mean    StdDev    Median
-------------------------------------------------------------------------------------------
bench_encode_small_64                      0.0012    0.0050    0.0015    0.0003    0.0014
bench_encode_medium_1024                   0.0089    0.0156    0.0095    0.0012    0.0093
bench_encode_large_8192                    0.0672    0.0891    0.0712    0.0045    0.0701
-------------------------------------------------------------------------------------------
```

**Key metrics:**
- **Min**: Fastest execution time (best case)
- **Max**: Slowest execution time (worst case)
- **Mean**: Average execution time
- **StdDev**: Standard deviation (measure of consistency)
- **Median**: Middle value (less sensitive to outliers)

### Performance Targets

Based on initial testing, the following performance targets should be maintained:

| Operation | Target Speedup | Warning Threshold | Critical Threshold |
|-----------|---------------|-------------------|-------------------|
| encode() vs NumPy | 1.5-2.5x | < 1.3x | < 1.0x |
| encode_batch() vs NumPy | 2.0-4.0x | < 1.7x | < 1.2x |
| Scalability | Linear | > 1.3x expected | > 1.5x expected |

### Regression Detection

The CI/CD workflow automatically detects performance regressions:
- **Warning**: 10% slowdown from baseline
- **Critical**: 25% slowdown from baseline
- **Block**: 50% slowdown from baseline

## Continuous Integration

The `.github/workflows/benchmark.yml` workflow:

1. Runs on every push to `main` branch
2. Runs on every pull request
3. Stores benchmark results as artifacts
4. Compares against stored baseline
5. Fails build on critical regressions
6. Comments on PRs with performance comparison

## Best Practices

### For Development

1. **Run benchmarks before committing** significant changes
2. **Use consistent hardware** for comparisons (same machine, similar load)
3. **Warm up the system** with a few iterations before measuring
4. **Run multiple iterations** to account for variance
5. **Check for statistical significance**, not just means

### For Interpreting Results

1. **Look at median, not mean** (median is more robust to outliers)
2. **Check standard deviation** - high variance indicates unreliable measurements
3. **Compare multiple runs** to ensure consistency
4. **Consider real-world impact** - micro-optimizations may not matter
5. **Profile first** - use benchmarks to validate, not discover, bottlenecks

## Troubleshooting

### High Variance in Results

**Cause**: Background processes, thermal throttling, CPU frequency scaling

**Solutions:**
- Close unnecessary applications
- Run multiple times and check consistency
- Use `--benchmark-warmup` for JIT-compiled code
- Increase `--benchmark-min-rounds` for better statistics

### Inconsistent Comparisons

**Cause**: Different system states between runs

**Solutions:**
- Use `--benchmark-autosave` to store results
- Compare using `--benchmark-compare` with saved files
- Run both versions in same session for direct comparison

### Poor Performance

**Cause**: Not using release build, SIMD not enabled

**Solutions:**
- Ensure package built with `maturin develop --release`
- Check CPU supports required SIMD instructions
- Verify optimization flags in `Cargo.toml`

## Performance Analysis

### Expected Performance Characteristics

**Single encoding (encode()):**
- O(n) complexity where n = min(data_size, n_qubits)
- Minimal overhead for small arrays
- Better cache locality for medium arrays
- Constant time when data_size < n_qubits

**Batch encoding (encode_batch()):**
- O(batch_size * min(data_dim, n_qubits)) complexity
- Better SIMD utilization with larger batches
- Amortizes Python overhead over many operations
- Linear scaling with batch size

**SIMD Advantages:**
- 2-4x speedup over NumPy for batch operations
- 1.5-2.5x speedup for single operations
- More consistent performance (lower variance)
- Better cache efficiency with chunked processing

## Contributing

When adding new benchmarks:

1. **Use descriptive names** that indicate what's being tested
2. **Include multiple scenarios** (small, medium, large)
3. **Add docstrings** explaining the benchmark purpose
4. **Set appropriate round counts** based on operation duration
5. **Document expected performance** in comments
6. **Update this README** with new benchmark categories

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [criterion.rs documentation](https://bheisler.github.io/criterion.rs/book/)
- [Python profiling guide](https://docs.python.org/3/library/profile.html)
- [Rust performance guide](https://nnethercote.github.io/perf-book/)

## License

MIT License - See main project LICENSE file
