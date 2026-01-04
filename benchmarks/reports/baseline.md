# SIMD Angle Encoder - Phase 1 Baseline Performance Report

**Generated**: 2026-01-04T06:48:24.029600
**Commit**: `27812838`
**Branch**: main

## Test Environment

- **System**: Darwin arm
- **CPU**: Apple M3 Pro (12 cores)
- **Python**: 3.12.5

## Executive Summary

**Total Benchmarks Executed**: 189
**Benchmark Categories**: 24

### Benchmark Categories

- batch-encode-dimensions
- batch-encode-edge
- batch-encode-mixed
- batch-encode-qubits
- batch-encode-scaling
- batch-encode-single
- batch-encode-size
- batch-encode-throughput
- comparative-batch
- comparative-scaling
- comparative-single
- comparative-speedup
- encode-edge
- encode-qubits
- encode-repeated
- encode-scaling
- encode-single
- scalability-batch-size
- scalability-data-size
- scalability-linearity
- scalability-memory
- scalability-mixed
- scalability-real-world
- scalability-throughput

## Performance Analysis: SIMD vs NumPy

| Data Size | NumPy Time (μs) | SIMD Time (μs) | Speedup Factor | Improvement |
|-----------|-----------------|---------------|----------------|-------------|
| null       |        2836.94 |         30.00 |          94.57x |       98.94% |
| 1          |           8.41 |          0.61 |          13.78x |       92.74% |
| 4          |           0.66 |          0.38 |           1.74x |       42.47% |
| 8          |           0.91 |          0.38 |           2.41x |       58.49% |
| 10         |          80.64 |          1.81 |          44.49x |       97.75% |
| 16         |           1.56 |          0.32 |           4.86x |       79.44% |
| 32         |           2.87 |          0.33 |           8.60x |       88.38% |
| 50         |         399.65 |          7.10 |          56.28x |       98.22% |
| 64         |           5.51 |          0.46 |          11.93x |       91.62% |
| 100        |         798.43 |         14.13 |          56.49x |       98.23% |
| 128        |          10.76 |          0.43 |          25.28x |       96.04% |
| 256        |          21.06 |          0.51 |          41.43x |       97.59% |
| 500        |        4050.87 |         63.13 |          64.17x |       98.44% |
| 512        |          44.26 |          0.64 |          68.79x |       98.55% |
| 1000       |        8079.31 |        133.93 |          60.33x |       98.34% |
| 1024       |          88.60 |          0.92 |          95.81x |       98.96% |

**Summary Statistics:**
- **Average Speedup**: 40.69x
- **Maximum Speedup**: 95.81x
- **Minimum Speedup**: 1.74x

## Scalability Analysis

### Scalability Batch Size

**Scaling Ratio**: 1.11 (1.0 = perfect linear scaling)
**Linear Scaling**: ✅ Yes

| Data Size | Time (ms) | Throughput (ops/ms) |
|-----------|-----------|---------------------|
|         1 |    0.0006 |              1649.54 |
|         5 |    0.0011 |              4706.11 |
|         8 |    0.0055 |              1452.90 |
|        10 |    0.0017 |              5759.26 |
|        16 |    0.0063 |              2530.60 |
|        25 |    0.0040 |              6268.79 |
|        32 |    0.0074 |              4335.61 |
|        50 |    0.0077 |              6525.00 |
|        64 |    0.0150 |              4279.42 |
|       100 |    0.0140 |              7135.77 |
|       128 |    0.0169 |              7553.21 |
|       250 |    0.0327 |              7637.32 |
|       256 |    0.0321 |              7965.77 |
|       500 |    0.0648 |              7711.01 |
|       512 |    0.0502 |             10194.82 |
|      1000 |    0.1292 |              7740.40 |
|      1024 |    0.0967 |             10586.03 |
|      2500 |    0.3223 |              7757.58 |
|      5000 |    0.7915 |              6316.96 |
|     10000 |    1.5520 |              6443.16 |

### Scalability Data Size

**Scaling Ratio**: 0.80 (1.0 = perfect linear scaling)
**Linear Scaling**: ❌ No

| Data Size | Time (ms) | Throughput (ops/ms) |
|-----------|-----------|---------------------|
|         4 |    0.0004 |             10160.64 |
|         4 |    0.0004 |              9047.35 |
|         8 |    0.0003 |             23254.33 |
|         8 |    0.0004 |             20416.41 |
|        16 |    0.0003 |             45873.02 |
|        16 |    0.0004 |             43070.44 |
|        32 |    0.0004 |             87399.90 |
|        32 |    0.0004 |             80645.19 |
|        64 |    0.0005 |            141973.49 |
|        64 |    0.0005 |            131134.27 |
|       128 |    0.0005 |            265500.79 |
|       128 |    0.0005 |            277351.05 |
|       256 |    0.0005 |            475130.73 |
|       256 |    0.0005 |            503502.92 |
|       512 |    0.0007 |            763436.31 |
|      1024 |    0.0010 |           1065696.08 |
|      2048 |    0.0016 |           1291184.70 |
|      4096 |    0.0031 |           1340580.24 |

### Scalability Linearity

**Scaling Ratio**: 68.56 (1.0 = perfect linear scaling)
**Linear Scaling**: ❌ No

| Data Size | Time (ms) | Throughput (ops/ms) |
|-----------|-----------|---------------------|
|         1 |    0.0004 |              2410.95 |
|         1 |    0.0155 |                64.48 |
|         2 |    0.0004 |              4716.41 |
|         2 |    0.0262 |                76.34 |
|         4 |    0.0006 |              7176.21 |
|         4 |    0.0554 |                72.15 |
|         8 |    0.0006 |             12419.22 |
|         8 |    0.1099 |                72.76 |
|        16 |    0.0009 |             16857.89 |
|        16 |    0.2351 |                68.07 |

### Scalability Mixed

**Scaling Ratio**: 1.58 (1.0 = perfect linear scaling)
**Linear Scaling**: ❌ No

| Data Size | Time (ms) | Throughput (ops/ms) |
|-----------|-----------|---------------------|
|        10 |    0.0010 |              9699.94 |
|        50 |    0.0040 |             12473.51 |
|       100 |    0.0152 |              6565.02 |
|       500 |    0.0817 |              6120.45 |
|      1000 |    0.3964 |              2522.97 |
|      1000 |    1.5022 |               665.71 |
|      5000 |    1.7313 |              2888.05 |
|      5000 |    3.7329 |              1339.43 |
|     10000 |    1.9968 |              5008.05 |

### Scalability Throughput

**Scaling Ratio**: 1.30 (1.0 = perfect linear scaling)
**Linear Scaling**: ❌ No

| Data Size | Time (ms) | Throughput (ops/ms) |
|-----------|-----------|---------------------|
|       100 |    0.0149 |              6722.15 |
|       500 |    0.0847 |              5901.07 |
|      1000 |    0.1728 |              5786.89 |
|      5000 |    1.1130 |              4492.41 |
|     10000 |    3.9434 |              2535.86 |

## Phase 2 Performance Targets

### Micro-Benchmark Targets

Target: 20% improvement across all single operations

| Benchmark | Baseline (μs) | Target (μs) |
|-----------|---------------|-------------|
| test_encode_single_value                 |        0.3742 |      0.2994 |
| test_encode_tiny_data_4                  |        0.3113 |      0.2490 |
| test_encode_small_data_16                |        0.3207 |      0.2565 |
| test_encode_medium_data_64               |        0.4009 |      0.3207 |
| test_encode_large_data_256               |        0.5109 |      0.4087 |
| ... and 6 more | | |

### Comparative Performance Targets

Target: 1.5x improvement in speedup factor OR minimum 50x speedup

| Data Size | Current Speedup | Target Speedup | Improvement Needed |
|-----------|-----------------|----------------|-------------------|
| null       |           94.57x |         141.85x | 50.0%             |
| 1          |           13.78x |          50.00x | 262.8%            |
| 4          |            1.74x |          50.00x | 2776.7%           |
| 8          |            2.41x |          50.00x | 1975.6%           |
| 10         |           44.49x |          66.73x | 50.0%             |
| 16         |            4.86x |          50.00x | 928.0%            |
| 32         |            8.60x |          50.00x | 481.1%            |
| 50         |           56.28x |          84.42x | 50.0%             |
| 64         |           11.93x |          50.00x | 319.2%            |
| 100        |           56.49x |          84.74x | 50.0%             |
| 128        |           25.28x |          50.00x | 97.8%             |
| 256        |           41.43x |          62.15x | 50.0%             |
| 500        |           64.17x |          96.25x | 50.0%             |
| 512        |           68.79x |         103.18x | 50.0%             |
| 1000       |           60.33x |          90.49x | 50.0%             |
| 1024       |           95.81x |         143.72x | 50.0%             |

## Detailed Benchmark Results

### Batch Encode Dimensions

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_dim_8                            |    5.5932 |   5.0830 |  14.4170 |      0.2605 |      5.5830 | 109589 |
| test_encode_batch_dim_16                           |    6.2384 |   5.5830 |  52.9580 |      0.4177 |      6.2080 | 134085 |
| test_encode_batch_dim_32                           |    7.4920 |   6.7080 |  16.1670 |      0.3695 |      7.4580 |  84211 |
| test_encode_batch_dim_64                           |   13.5476 |  12.5000 | 235.0410 |      1.3148 |     13.3340 |  47153 |
| test_encode_batch_dim_128                          |   17.4732 |  16.5830 |  27.5000 |      0.7055 |     17.3750 |  26003 |
| test_encode_batch_dim_256                          |   28.1354 |  27.0000 |  43.3340 |      1.0790 |     27.9170 |  15707 |
| test_encode_batch_dim_512                          |   49.4754 |  48.5000 |  62.8330 |      1.5407 |     48.9170 |   8674 |

### Batch Encode Edge

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_dim_smaller_than_qubits           |    3.2219 |   2.8750 |  16.4160 |      0.2518 |      3.2080 | 187477 |
| test_encode_batch_dim_larger_than_qubits           |    3.7133 |   3.4165 |   8.2500 |      0.1080 |      3.7080 | 134826 |
| test_encode_batch_zeros                            |    3.9600 |   3.5840 |  11.7500 |      0.1392 |      3.9580 | 147233 |
| test_encode_batch_ones                             |    3.9900 |   3.5625 |  34.4795 |      0.2514 |      3.9375 | 130430 |

### Batch Encode Mixed

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_small_batch_large_dim            |    3.4128 |   2.9160 |  35.0830 |      0.3452 |      3.3750 | 163266 |
| test_encode_batch_large_batch_small_dim            |   56.5848 |  44.0840 | 634.4590 |     14.8415 |     52.0830 |   4421 |
| test_encode_batch_balanced                         |   22.9774 |  16.5000 | 1969.0410 |     16.1272 |     18.2920 |  36867 |

### Batch Encode Qubits

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_qubits_4                         |    5.8028 |   5.5830 |  16.2500 |      0.3391 |      5.7500 | 141164 |
| test_encode_batch_qubits_8                         |    6.0666 |   5.8750 |  16.0830 |      0.3430 |      6.0000 | 127665 |
| test_encode_batch_qubits_16                        |    6.5831 |   5.7920 | 392.4160 |      1.6094 |      6.5830 | 125660 |
| test_encode_batch_qubits_32                        |   10.2115 |   9.0000 |  27.5000 |      0.4611 |     10.1670 |  63328 |
| test_encode_batch_qubits_64                        |   15.5030 |  11.7080 |  43.8330 |      5.2691 |     12.8340 |  10806 |

### Batch Encode Scaling

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_scaling[10-16]         [10-16]   |    1.0207 |   0.9125 |   2.5729 |      0.0371 |      1.0167 |  50633 |
| test_encode_batch_scaling[50-32]         [50-32]   |    4.0021 |   3.6660 |  10.7090 |      0.1451 |      4.0000 | 175194 |
| test_encode_batch_scaling[100-64]        [100-64]  |   15.0608 |  13.6250 | 143.2920 |      3.3546 |     14.5000 |  49483 |
| test_encode_batch_scaling[500-128]       [500-128] |   88.9426 |  74.7500 | 847.3750 |     14.2833 |     85.3340 |   7574 |
| test_encode_batch_scaling[1000-256]      [1000-256] |  387.8938 | 343.0420 | 656.0420 |     23.8552 |    382.4590 |   1971 |
| test_encode_batch_scaling[5000-256]      [5000-256] | 1692.1809 | 1577.8340 | 3110.4580 |    116.7217 |   1668.9790 |    380 |

### Batch Encode Single

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_single_row                       |    0.4579 |   0.3750 |   8.7080 |      0.0754 |      0.4580 |  65219 |
| test_encode_batch_small_8x4                        |    0.7684 |   0.7333 |   1.7208 |      0.0309 |      0.7625 | 120006 |
| test_encode_batch_medium_32x16                     |    2.3619 |   2.2500 |   7.1040 |      0.1051 |      2.3540 | 180440 |
| test_encode_batch_large_128x32                     |    9.7969 |   9.3330 |  19.2500 |      0.3746 |      9.7500 |  68772 |

### Batch Encode Size

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_size_1                           |    0.4102 |   0.3854 |   1.2223 |      0.0279 |      0.4063 | 195123 |
| test_encode_batch_size_10                          |    1.1065 |   1.0582 |   2.8750 |      0.0479 |      1.1000 | 176492 |
| test_encode_batch_size_50                          |    7.1475 |   6.7920 |  16.2090 |      0.3446 |      7.1250 |  67413 |
| test_encode_batch_size_100                         |   13.7674 |  12.9590 | 115.1670 |      2.6764 |     13.4170 |  39867 |
| test_encode_batch_size_500                         |   71.1316 |  58.3330 | 667.4590 |     12.1010 |     70.5410 |   9192 |
| test_encode_batch_size_1000                        |  140.8251 | 120.8340 | 556.3750 |      8.7494 |    140.8750 |   5453 |
| test_encode_batch_size_5000                        |  866.2518 | 742.7920 | 2066.4580 |    101.8001 |    851.0420 |    879 |
| test_encode_batch_size_10000                       | 1637.0258 | 1522.6250 | 2713.5420 |     80.3132 |   1626.0415 |    546 |

### Batch Encode Throughput

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_throughput_small                 |    7.4274 |   6.6660 |  16.9580 |      0.3149 |      7.3750 |  98368 |
| test_encode_batch_throughput_medium                |   68.1850 |  61.8330 | 377.0830 |      7.4707 |     66.2090 |  10802 |
| test_encode_batch_throughput_large                 | 1245.7411 | 1022.2920 | 4852.2920 |    287.9975 |   1197.4790 |    622 |

### Comparative Batch

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_numpy_baseline_batch_small                    |   21.1918 |  20.6250 |  80.7090 |      0.9261 |     21.0000 |  36037 |
| test_simd_batch_small                              |    1.0620 |   0.9580 |   9.0000 |      0.0697 |      1.0420 | 195122 |
| test_numpy_baseline_batch_medium                   |  802.3877 | 747.1250 | 2641.4580 |     53.9226 |    798.3330 |   1227 |
| test_simd_batch_medium                             |   14.9297 |  13.8330 |  25.7500 |      0.6163 |     14.7920 |  27875 |
| test_numpy_baseline_batch_large                    | 16169.6412 | 16065.6670 | 16462.5830 |     98.0992 |  16145.0835 |     62 |
| test_simd_batch_large                              |  162.6917 | 152.6250 | 471.1660 |      9.4228 |    160.6670 |   3724 |

### Comparative Scaling

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_numpy_encode_scaling[4]             [4]       |    0.6622 |   0.5830 |  32.1250 |      0.1566 |      0.6660 | 198334 |
| test_numpy_encode_scaling[8]             [8]       |    0.9133 |   0.8833 |   2.6875 |      0.0418 |      0.9042 | 104341 |
| test_numpy_encode_scaling[16]            [16]      |    1.5609 |   1.4792 |  13.7918 |      0.0860 |      1.5520 | 164393 |
| test_numpy_encode_scaling[32]            [32]      |    2.8739 |   2.7915 |  41.8540 |      0.1853 |      2.8540 | 164393 |
| test_numpy_encode_scaling[64]            [64]      |    5.5114 |   5.1660 | 140.7090 |      0.5526 |      5.4580 | 126985 |
| test_numpy_encode_scaling[128]           [128]     |   10.7553 |   9.8330 | 111.2500 |      0.8249 |     10.6250 |  66116 |
| test_numpy_encode_scaling[256]           [256]     |   21.0577 |  19.4580 |  61.6250 |      0.8861 |     20.8750 |  42404 |
| test_numpy_encode_scaling[512]           [512]     |   44.2580 |  40.3330 |  99.8330 |      1.5917 |     43.8330 |  21939 |
| test_numpy_encode_scaling[1024]          [1024]    |   88.5983 |  82.2500 | 110.7500 |      2.4832 |     87.7080 |  11788 |
| test_simd_encode_scaling[4]              [4]       |    0.3810 |   0.2500 |   9.9170 |      0.0859 |      0.3750 |  65933 |
| test_simd_encode_scaling[8]              [8]       |    0.3791 |   0.2910 |   9.1250 |      0.0713 |      0.3750 | 167842 |
| test_simd_encode_scaling[16]             [16]      |    0.3209 |   0.2889 |   2.5805 |      0.0269 |      0.3167 | 196696 |
| test_simd_encode_scaling[32]             [32]      |    0.3340 |   0.3125 |   0.9854 |      0.0199 |      0.3312 | 143720 |
| test_simd_encode_scaling[64]             [64]      |    0.4621 |   0.3750 |   8.9580 |      0.0851 |      0.4580 | 114286 |
| test_simd_encode_scaling[128]            [128]     |    0.4254 |   0.3750 |   2.8055 |      0.0307 |      0.4202 | 190477 |
| test_simd_encode_scaling[256]            [256]     |    0.5083 |   0.4708 |   1.4667 |      0.0289 |      0.5041 | 171439 |
| test_simd_encode_scaling[512]            [512]     |    0.6434 |   0.5750 |  20.9500 |      0.1134 |      0.6333 | 134826 |
| test_simd_encode_scaling[1024]           [1024]    |    0.9247 |   0.8458 |   3.5562 |      0.0622 |      0.9083 |  56073 |
| test_numpy_batch_scaling[1]              [1]       |    8.4084 |   8.0830 | 107.7500 |      0.5942 |      8.3330 |  71639 |
| test_numpy_batch_scaling[10]             [10]      |   80.6445 |  75.2090 | 107.2920 |      2.3560 |     79.7500 |  12067 |
| ... and 10 more | | | | | | |

### Comparative Single

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_numpy_baseline_small                          |    1.7505 |   1.5000 |  63.8750 |      0.6073 |      1.7080 |  28005 |
| test_simd_small                                    |    0.3346 |   0.2891 |  36.4375 |      0.1307 |      0.3307 | 195123 |
| test_numpy_baseline_medium                         |    5.6470 |   5.2500 |  36.9590 |      0.3507 |      5.5830 |  86648 |
| test_simd_medium                                   |    0.4744 |   0.3750 |   2.5830 |      0.0344 |      0.4580 | 127649 |
| test_numpy_baseline_large                          |   21.0409 |  19.8330 | 124.4580 |      0.9735 |     20.8340 |  40747 |
| test_simd_large                                    |    0.5009 |   0.4667 |   1.6917 |      0.0229 |      0.5000 | 169033 |

### Comparative Speedup

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_speedup_small_data_64                         |    0.4169 |   0.3646 |   9.4132 |      0.0957 |      0.4063 | 192013 |
| test_speedup_medium_data_1024                      |    0.9119 |   0.8271 |   7.5271 |      0.0652 |      0.9020 |  54921 |
| test_speedup_large_data_8192                       |   10.9734 |   4.8340 |  38.5000 |      4.4756 |     13.8750 |  22663 |
| test_speedup_batch_small                           |    1.0066 |   0.9250 |  22.3000 |      0.1328 |      0.9916 | 181818 |
| test_speedup_batch_medium                          |   15.2829 |  14.3750 |  30.8340 |      0.7160 |     15.1250 |  58535 |
| test_speedup_batch_large                           |  166.9253 | 156.1670 | 1142.4580 |     24.1234 |    162.9170 |   3588 |

### Encode Edge

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_data_smaller_than_qubits               |    0.3807 |   0.2910 |  12.7080 |      0.0630 |      0.3750 | 198334 |
| test_encode_data_larger_than_qubits                |    0.3415 |   0.3036 |   5.5149 |      0.0421 |      0.3333 | 192013 |
| test_encode_zeros                                  |    0.4077 |   0.3507 |  17.7812 |      0.1198 |      0.3889 | 198335 |
| test_encode_ones                                   |    0.4043 |   0.3482 |  22.6071 |      0.0852 |      0.3929 | 186047 |
| test_encode_mixed_values                           |    0.3823 |   0.2910 |   8.9160 |      0.1499 |      0.3750 | 166668 |

### Encode Qubits

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_4_qubits                               |    0.3193 |   0.2768 |   0.7262 |      0.0165 |      0.3184 | 175194 |
| test_encode_8_qubits                               |    0.3782 |   0.2910 |   4.4160 |      0.0369 |      0.3750 | 192012 |
| test_encode_16_qubits                              |    0.3423 |   0.3021 |   2.8208 |      0.0208 |      0.3416 | 137932 |
| test_encode_32_qubits                              |    0.3618 |   0.3083 |   1.5875 |      0.0206 |      0.3604 | 139529 |
| test_encode_64_qubits                              |    0.4134 |   0.3472 |   3.2848 |      0.0267 |      0.4132 | 183218 |

### Encode Repeated

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_repeated_small                         |    7.3927 |   6.4580 |  26.6250 |      0.7681 |      7.2500 |  64342 |
| test_encode_repeated_medium                        |    9.3834 |   8.1670 |  68.3750 |      0.9564 |      9.1250 |  74075 |
| test_encode_repeated_large                         |   14.7938 |  13.4170 |  38.5000 |      1.1062 |     14.4580 |  53334 |

### Encode Scaling

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_scaling[8]                   [8]       |    0.3098 |   0.2696 |   1.7304 |      0.0365 |      0.3015 | 193537 |
| test_encode_scaling[16]                  [16]      |    0.3259 |   0.2818 |   2.1593 |      0.0236 |      0.3186 | 188965 |
| test_encode_scaling[32]                  [32]      |    0.3393 |   0.2945 |  12.8500 |      0.0647 |      0.3361 | 196696 |
| test_encode_scaling[64]                  [64]      |    0.3886 |   0.3590 |   1.2051 |      0.0190 |      0.3878 | 196735 |
| test_encode_scaling[128]                 [128]     |    0.4858 |   0.3750 |   5.7080 |      0.0465 |      0.5000 |  76191 |
| test_encode_scaling[256]                 [256]     |    0.4749 |   0.4280 |   1.2007 |      0.0188 |      0.4735 | 192012 |
| test_encode_scaling[512]                 [512]     |    0.6311 |   0.5666 |   3.5208 |      0.0292 |      0.6250 | 152859 |
| test_encode_scaling[1024]                [1024]    |    0.9861 |   0.8750 |  11.0420 |      0.0747 |      0.9590 | 114286 |

### Encode Single

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_single_value                           |    0.3742 |   0.2910 |   9.4580 |      0.1306 |      0.3750 |  57831 |
| test_encode_tiny_data_4                            |    0.3113 |   0.2734 |   4.0938 |      0.0343 |      0.3073 | 190476 |
| test_encode_small_data_16                          |    0.3207 |   0.2734 |   3.7864 |      0.0351 |      0.3151 | 200000 |
| test_encode_medium_data_64                         |    0.4009 |   0.3416 |   3.3729 |      0.0401 |      0.3896 | 129033 |
| test_encode_large_data_256                         |    0.5109 |   0.4334 |  35.7792 |      0.0993 |      0.5042 | 172682 |
| test_encode_xlarge_data_1024                       |    0.9433 |   0.8438 |   1.3750 |      0.0348 |      0.9375 |  55945 |
| test_encode_xxlarge_data_4096                      |    3.1008 |   2.5000 |   9.0420 |      0.2779 |      3.0830 | 112676 |

### Scalability Batch Size

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_batch_size_scaling[1]               [1]       |    0.6062 |   0.5000 |   9.4580 |      0.0932 |      0.5840 |  61226 |
| test_batch_size_scaling[5]               [5]       |    1.0624 |   1.0084 |   4.1166 |      0.0569 |      1.0500 | 173914 |
| test_batch_size_scaling[10]              [10]      |    1.7363 |   1.6387 |  11.2917 |      0.1012 |      1.7083 | 190476 |
| test_batch_size_scaling[25]              [25]      |    3.9880 |   3.8330 |  17.3750 |      0.1977 |      3.9580 | 117647 |
| test_batch_size_scaling[50]              [50]      |    7.6628 |   6.9160 |  30.7500 |      0.4800 |      7.7080 |  76430 |
| test_batch_size_scaling[100]             [100]     |   14.0139 |  13.4170 |  28.3340 |      0.7054 |     13.8750 |  44364 |
| test_batch_size_scaling[250]             [250]     |   32.7340 |  32.1250 |  50.0000 |      1.1332 |     32.4170 |  20943 |
| test_batch_size_scaling[500]             [500]     |   64.8424 |  63.6670 |  83.7090 |      1.9125 |     64.1660 |  10417 |
| test_batch_size_scaling[1000]            [1000]    |  129.1923 | 127.0830 | 166.1660 |      3.3535 |    127.7920 |   4988 |
| test_batch_size_scaling[2500]            [2500]    |  322.2654 | 317.3750 | 359.0000 |      5.0504 |    320.7500 |   2068 |
| test_batch_size_scaling[5000]            [5000]    |  791.5204 | 742.9170 | 1272.2500 |     42.8619 |    782.0830 |    977 |
| test_batch_size_scaling[10000]           [10000]   | 1552.0324 | 1496.4170 | 1808.0830 |     48.8582 |   1539.8335 |    534 |
| test_batch_dimension_scaling[8]          [8]       |    5.5062 |   5.1660 |  15.9580 |      0.2730 |      5.4580 | 133334 |
| test_batch_dimension_scaling[16]         [16]      |    6.3226 |   5.7500 | 172.2080 |      0.6149 |      6.2500 | 118232 |
| test_batch_dimension_scaling[32]         [32]      |    7.3807 |   7.0000 |  19.7500 |      0.3391 |      7.3330 | 105720 |
| test_batch_dimension_scaling[64]         [64]      |   14.9553 |  14.4160 |  30.0830 |      0.6357 |     14.8330 |  40269 |
| test_batch_dimension_scaling[128]        [128]     |   16.9464 |  16.3750 |  32.0420 |      0.6850 |     16.8330 |  31373 |
| test_batch_dimension_scaling[256]        [256]     |   32.1375 |  27.2910 | 483.5830 |      9.0686 |     28.2920 |  17083 |
| test_batch_dimension_scaling[512]        [512]     |   50.2216 |  48.7500 |  65.9170 |      1.6053 |     49.7080 |  10051 |
| test_batch_dimension_scaling[1024]       [1024]    |   96.7313 |  92.3750 | 224.7080 |      7.1339 |     94.5000 |   4451 |

### Scalability Data Size

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_data_size_scaling[4]         [4]       |    0.3937 |   0.2920 |   9.0420 |      0.0575 |      0.3750 |  55945 |
| test_encode_data_size_scaling[8]         [8]       |    0.3440 |   0.3125 |   2.7946 |      0.0232 |      0.3393 | 193536 |
| test_encode_data_size_scaling[16]        [16]      |    0.3488 |   0.3036 |   4.1399 |      0.0333 |      0.3423 | 200000 |
| test_encode_data_size_scaling[32]        [32]      |    0.3661 |   0.3066 | 506.4196 |      1.1752 |      0.3541 | 193574 |
| test_encode_data_size_scaling[64]        [64]      |    0.4508 |   0.4015 |   1.8977 |      0.0224 |      0.4470 | 191976 |
| test_encode_data_size_scaling[128]       [128]     |    0.4821 |   0.3750 |   7.5410 |      0.0511 |      0.4590 | 121213 |
| test_encode_data_size_scaling[256]       [256]     |    0.5388 |   0.4958 |   1.6625 |      0.0277 |      0.5333 | 175162 |
| test_encode_data_size_scaling[512]       [512]     |    0.6707 |   0.6083 |   4.5334 |      0.0384 |      0.6666 | 130430 |
| test_encode_data_size_scaling[1024]      [1024]    |    0.9609 |   0.9000 |   3.0834 |      0.0451 |      0.9582 | 200000 |
| test_encode_data_size_scaling[2048]      [2048]    |    1.5861 |   1.4583 |  97.2082 |      0.4767 |      1.5625 | 149993 |
| test_encode_data_size_scaling[4096]      [4096]    |    3.0554 |   2.5420 |  34.0830 |      0.3191 |      3.0410 | 107620 |
| test_encode_qubit_scaling[4]             [4]       |    0.4421 |   0.3330 |  71.5830 |      0.2691 |      0.4170 |  72512 |
| test_encode_qubit_scaling[8]             [8]       |    0.3918 |   0.3269 |   5.6122 |      0.0326 |      0.3878 | 190476 |
| test_encode_qubit_scaling[16]            [16]      |    0.3715 |   0.3333 |   1.0865 |      0.0251 |      0.3654 | 196696 |
| test_encode_qubit_scaling[32]            [32]      |    0.3968 |   0.3493 |  25.7051 |      0.0840 |      0.3910 | 191976 |
| test_encode_qubit_scaling[64]            [64]      |    0.4880 |   0.3750 |  10.0420 |      0.0766 |      0.5000 |  39933 |
| test_encode_qubit_scaling[128]           [128]     |    0.4615 |   0.4167 |   1.5250 |      0.0271 |      0.4583 | 200000 |
| test_encode_qubit_scaling[256]           [256]     |    0.5084 |   0.4708 |   1.4833 |      0.0276 |      0.5042 | 177778 |

### Scalability Linearity

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_linear_scaling[1]            [1]       |    0.4148 |   0.3507 |   3.9028 |      0.0341 |      0.4028 | 184604 |
| test_encode_linear_scaling[2]            [2]       |    0.4241 |   0.3958 |   1.3681 |      0.0220 |      0.4202 | 184639 |
| test_encode_linear_scaling[4]            [4]       |    0.5574 |   0.4580 |  11.2920 |      0.0646 |      0.5420 | 119403 |
| test_encode_linear_scaling[8]            [8]       |    0.6442 |   0.6083 |   2.1167 |      0.0319 |      0.6416 | 146349 |
| test_encode_linear_scaling[16]           [16]      |    0.9491 |   0.9104 |   1.5458 |      0.0432 |      0.9396 |  51838 |
| test_batch_linear_scaling[1]             [1]       |   15.5093 |  12.6660 | 191.0840 |      6.7704 |     13.6670 |  62337 |
| test_batch_linear_scaling[2]             [2]       |   26.1984 |  25.5410 |  89.8750 |      1.4860 |     25.8330 |  36037 |
| test_batch_linear_scaling[4]             [4]       |   55.4433 |  50.2500 | 112.7920 |      4.7484 |     55.1670 |  15842 |
| test_batch_linear_scaling[8]             [8]       |  109.9438 |  94.7920 | 417.4580 |     10.1288 |    108.8330 |   7862 |
| test_batch_linear_scaling[16]            [16]      |  235.0661 | 222.5830 | 380.5830 |     11.4934 |    235.3750 |   3775 |

### Scalability Memory

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_small_memory_footprint                 |    0.3318 |   0.2994 |   1.0052 |      0.0167 |      0.3307 | 198334 |
| test_encode_medium_memory_footprint                |    0.5091 |   0.4417 |   4.1750 |      0.0317 |      0.5042 | 184605 |
| test_encode_large_memory_footprint                 |    3.1125 |   2.6670 |   9.0000 |      0.2531 |      3.1250 |  79739 |
| test_encode_very_large_memory_footprint            |   30.0181 |   9.7500 | 216.2080 |     17.9945 |     27.4590 |  32259 |
| test_batch_cache_friendly                          |  147.7199 | 143.0410 | 205.7920 |      4.2720 |    146.5830 |   6213 |
| test_batch_cache_unfriendly                        | 3936.8977 | 3735.7080 | 4672.2920 |    142.2026 |   3898.0415 |    224 |

### Scalability Mixed

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_batch_mixed_scaling[10-16]          [10-16]   |    1.0309 |   0.9666 |   7.5582 |      0.0652 |      1.0250 | 186047 |
| test_batch_mixed_scaling[50-32]          [50-32]   |    4.0085 |   3.5420 |  20.8750 |      0.2261 |      4.0000 | 161057 |
| test_batch_mixed_scaling[100-64]         [100-64]  |   15.2322 |  14.5830 |  28.6660 |      0.6838 |     15.0420 |  21486 |
| test_batch_mixed_scaling[500-128]        [500-128] |   81.6934 |  79.4580 | 105.9170 |      2.4701 |     80.9580 |   6975 |
| test_batch_mixed_scaling[1000-256]       [1000-256] |  396.3580 | 349.5000 | 1073.6670 |     42.7290 |    386.4370 |   1860 |
| test_batch_mixed_scaling[5000-256]       [5000-256] | 1731.2701 | 1602.5000 | 3055.2500 |    165.9858 |   1681.2500 |    183 |
| test_batch_mixed_scaling[10000-128]      [10000-128] | 1996.7857 | 1872.7500 | 3205.7080 |    133.8010 |   1957.1040 |    510 |
| test_batch_mixed_scaling[5000-512]       [5000-512] | 3732.9399 | 3517.3750 | 5396.0000 |    234.2645 |   3659.7500 |    113 |
| test_batch_mixed_scaling[1000-1024]      [1000-1024] | 1502.1544 | 1400.4590 | 1823.8750 |     68.5352 |   1481.0830 |    451 |

### Scalability Real World

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_vqc_small_training_batch                      |    3.9482 |   3.7080 |  14.0420 |      0.2202 |      3.9170 | 175194 |
| test_vqc_medium_training_batch                     |   18.8908 |  16.3330 | 390.0830 |      6.5968 |     16.7910 |  38462 |
| test_vqc_large_training_batch                      |   98.5523 |  74.9160 | 2350.6250 |     52.7551 |     90.0420 |   9597 |
| test_quantum_data_preprocessing                    |   39.9291 |  30.8330 | 2754.0830 |     68.2542 |     32.6250 |   1954 |
| test_inference_workload                            |   47.5924 |  45.8330 |  61.9580 |      1.5403 |     47.1670 |   9196 |

### Scalability Throughput

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_throughput_large_batches[100-64]    [100-64]  |   14.8762 |  13.7500 |  25.4580 |      0.6252 |     14.7500 |  12794 |
| test_throughput_large_batches[500-64]    [500-64]  |   84.7304 |  68.4590 | 334.6660 |     21.7126 |     74.8750 |  10900 |
| test_throughput_large_batches[1000-128]  [1000-128] |  172.8044 | 158.2500 | 277.4580 |     12.7127 |    168.6250 |   4756 |
| test_throughput_large_batches[5000-128]  [5000-128] | 1112.9879 | 1054.5420 | 1296.2910 |     43.8840 |   1099.4170 |    695 |
| test_throughput_large_batches[10000-256] [10000-256] | 3943.4280 | 3755.1670 | 4498.1670 |    133.9303 |   3896.2920 |    209 |

## Recommendations for Phase 2

### Performance Optimization Priorities

1. **High Priority**: Improve SIMD vectorization to achieve 50x+ speedup
   - Current average speedup: 40.69x
   - Target: 50-90x speedup for batch operations
   - Focus on: Memory alignment, cache optimization, instruction selection

2. **Medium Priority**: Optimize batch processing throughput
   - Implement batch-aware SIMD optimizations
   - Reduce per-batch overhead
   - Target: 10% improvement in throughput

3. **Low Priority**: Micro-optimizations for single operations
   - Target: 20% improvement
   - Focus on: Hot path optimization, branch reduction

### Testing & Validation

- Add continuous benchmarking in CI/CD pipeline
- Set regression thresholds: 10% warning, 25% critical
- Track performance trends across commits
- Validate performance on multiple platforms (x86_64, ARM64)

### Documentation

- Document performance characteristics in user guide
- Create performance tuning guide for different use cases
- Add benchmark results to project documentation

---

*This report serves as the Phase 1 baseline for all future performance comparisons.*
