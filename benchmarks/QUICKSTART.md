# Benchmark Quick Start Guide

Quick guide to running benchmarks for the SIMD Angle Encoder project.

## Prerequisites

1. **Build the Rust extension in release mode:**
   ```bash
   maturin develop --release
   ```

2. **Install benchmark dependencies:**
   ```bash
   pip install pytest pytest-benchmark numpy
   ```

## Quick Benchmark Runs

### Run All Benchmarks (Fast Mode)
```bash
# Using the convenience script
python benchmarks/run_benchmarks.py --fast

# Or using pytest directly
pytest benchmarks/ -v --benchmark-min-rounds=20
```

### Run All Benchmarks (Full Mode)
```bash
# Using the convenience script (recommended)
python benchmarks/run_benchmarks.py

# Or using pytest directly
pytest benchmarks/ -v --benchmark-min-rounds=50
```

### Run Specific Benchmark Categories

```bash
# Single encoding benchmarks
pytest benchmarks/python/test_bench_encode.py -v

# Batch encoding benchmarks
pytest benchmarks/python/test_bench_encode_batch.py -v

# Comparative benchmarks (vs NumPy)
pytest benchmarks/python/test_bench_comparative.py -v

# Scalability benchmarks
pytest benchmarks/python/test_bench_scalability.py -v
```

## Generate Reports

### Generate JSON and HTML Reports

```bash
# Generate JSON report
pytest benchmarks/ --benchmark-only --benchmark-json=reports/benchmark-results.json

# Generate HTML report
pytest benchmarks/ --benchmark-only --benchmark-html=reports/benchmark-results.html

# Generate both
pytest benchmarks/ --benchmark-only \
  --benchmark-json=reports/benchmark-results.json \
  --benchmark-html=reports/benchmark-results.html
```

### Generate Summary Report

```bash
# Using the convenience script
python benchmarks/run_benchmarks.py --report

# This generates:
# - reports/summary-report.json (JSON summary)
# - reports/summary-report.md (Markdown summary)
```

## Compare with Previous Runs

```bash
# Save current run as baseline
pytest benchmarks/ --benchmark-autosave --benchmark-save-data

# Compare against baseline
pytest benchmarks/ --benchmark-compare=0001 \
  --benchmark-compare-fail=mean:10% \
  --benchmark-compare-fail=median:15%
```

## Expected Performance

Based on initial testing on typical hardware:

| Operation | Data Size | Expected Time | vs NumPy |
|-----------|-----------|---------------|----------|
| encode() | 64 elements | ~0.01-0.02 ms | 1.5-2.5x faster |
| encode() | 1024 elements | ~0.05-0.10 ms | 1.5-2.5x faster |
| encode_batch() | 10x16 | ~0.05-0.15 ms | 1.5-2.0x faster |
| encode_batch() | 100x64 | ~0.5-1.5 ms | 2.0-4.0x faster |
| encode_batch() | 1000x128 | ~5-15 ms | 2.0-4.0x faster |

*Times are approximate and depend on hardware*

## Rust Benchmarks

```bash
# Run Rust criterion benchmarks
cargo bench

# Run specific benchmark group
cargo bench --bench angle_encode_bench encode_single

# Generate HTML report (auto-generated in target/criterion/)
cargo bench --bench angle_encode_bench
```

View criterion reports:
```bash
# Open in browser
open target/criterion/report/index.html  # macOS
xdg-open target/criterion/report/index.html  # Linux
```

## CI/CD Integration

Benchmarks run automatically on:
- Every push to `main` branch
- Every pull request
- Manual workflow dispatch

Results are stored as artifacts and:
- Compared against baseline for regression detection
- Posted as comments on PRs
- Failed builds on critical regressions (>25% slowdown)

## Performance Regression Detection

The workflow detects regressions at these thresholds:
- **Warning**: 10% slowdown ( informational )
- **Critical**: 25% slowdown (fails PR checks)
- **Block**: 50% slowdown (blocks merge)

## Troubleshooting

### Benchmarks not found
```bash
# Ensure package is built in release mode
maturin develop --release

# Verify import works
python -c "import simd_angle_encoder; print(simd_angle_encoder.simd_info())"
```

### High variance in results
```bash
# Increase iterations for better statistics
pytest benchmarks/ --benchmark-min-rounds=100

# Run with warmup
pytest benchmarks/ --benchmark-warmup

# Close other applications to reduce system noise
```

### Comparison fails
```bash
# Ensure you have saved baseline data
pytest benchmarks/ --benchmark-autosave

# Check saved files exist
ls benchmarks/reports/

# Use explicit comparison
pytest benchmarks/ --benchmark-compare=<baseline-id>
```

## Tips for Development

1. **Use fast mode during development:**
   ```bash
   python benchmarks/run_benchmarks.py --fast
   ```

2. **Run specific test categories you're working on:**
   ```bash
   pytest benchmarks/python/test_bench_encode.py::TestEncodeSingle -v
   ```

3. **Compare before and after your changes:**
   ```bash
   # Before
   pytest benchmarks/ --benchmark-autosave

   # Make changes...

   # After
   pytest benchmarks/ --benchmark-compare
   ```

4. **Profile if needed:**
   ```bash
   python -m cProfile -o profile.stats your_script.py
   python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"
   ```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/) for advanced options
- Review [criterion.rs book](https://bheisler.github.io/criterion.rs/book/) for Rust benchmarking
