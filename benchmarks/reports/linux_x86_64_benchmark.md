# SIMD Angle Encoder - Linux x86_64 Benchmark Report

**Date**: 2026-01-04  
**Platform**: Linux 6.8.0-1029-aws (x86_64)  
**Python**: 3.12.3  
**SIMD**: AVX-512

## Summary

| Metric | Value |
|--------|-------|
| **Average Speedup** | 70.96x |
| **Maximum Speedup** | 187.08x |
| **Minimum Speedup** | 3.50x |
| **SIMD Support** | AVX-512 |

## Single Encoding Performance

| Data Size | NumPy (μs) | SIMD (μs) | Speedup |
|-----------|------------|-----------|---------|
| 4 | 3.60 | 1.00 | 3.60x |
| 8 | 2.50 | 0.72 | 3.50x |
| 16 | 3.86 | 0.74 | 5.23x |
| 32 | 6.46 | 0.69 | 9.34x |
| 64 | 11.73 | 0.62 | 18.92x |
| 128 | 22.53 | 0.72 | 31.50x |
| 256 | 43.73 | 0.76 | 57.31x |
| 512 | 106.62 | 0.95 | 111.80x |
| 1024 | 211.83 | 1.14 | 185.10x |

## Batch Encoding Performance (data_size=128)

| Batch Size | NumPy (ms) | SIMD (ms) | Speedup |
|------------|------------|-----------|---------|
| 1 | 0.0223 | 0.0007 | 33.46x |
| 10 | 0.2230 | 0.0012 | 187.08x |
| 100 | 2.1552 | 0.0121 | 177.59x |
| 1000 | 21.6255 | 0.2205 | 98.06x |

## Key Findings

1. **AVX-512 delivers exceptional performance** - The 512-bit SIMD registers enable processing 16 floats simultaneously, resulting in speedups exceeding 180x for optimal workloads.

2. **Batch operations benefit most** - Peak speedup of 187.08x achieved with batch_size=10, demonstrating excellent vectorization efficiency.

3. **Large data scales well** - Single encoding speedup increases from 3.5x (8 elements) to 185x (1024 elements), showing near-linear scaling with data size.

4. **Sub-microsecond SIMD times** - SIMD encoding consistently completes in under 1.2μs regardless of data size up to 1024 elements.

## Comparison with macOS ARM (M3 Pro)

| Metric | Linux x86_64 (AVX-512) | macOS ARM (NEON) |
|--------|------------------------|------------------|
| Avg Speedup | 70.96x | 40.69x |
| Max Speedup | 187.08x | 95.81x |
| SIMD Width | 512-bit | 128-bit |

The AVX-512 implementation achieves ~1.7x higher average speedup compared to ARM NEON, primarily due to the 4x wider SIMD registers.
