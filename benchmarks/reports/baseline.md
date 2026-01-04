# SIMD Angle Encoder - Phase 1 Baseline Performance Report

**Generated**: 2026-01-04T20:07:09.403282
**Commit**: `b9a455e6`
**Branch**: main

## Test Environment

- **System**: Darwin arm
- **CPU**: Apple M3 Pro (12 cores)
- **Python**: 3.12.5

## Executive Summary

**Total Benchmarks Executed**: 121
**Benchmark Categories**: 17

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

## Performance Analysis: SIMD vs NumPy

| Data Size | NumPy Time (μs) | SIMD Time (μs) | Speedup Factor | Improvement |
|-----------|-----------------|---------------|----------------|-------------|
| null       |        2789.80 |         17.16 |         162.54x |       99.38% |
| 1          |           8.44 |          0.49 |          17.09x |       94.15% |
| 4          |           0.59 |          0.33 |           1.79x |       44.29% |
| 8          |           0.92 |          0.33 |           2.79x |       64.14% |
| 10         |          80.29 |          1.20 |          66.64x |       98.50% |
| 16         |           1.59 |          0.32 |           5.01x |       80.05% |
| 32         |           2.93 |          0.33 |           8.75x |       88.57% |
| 50         |         398.46 |          4.68 |          85.21x |       98.83% |
| 64         |           5.56 |          0.37 |          15.04x |       93.35% |
| 100        |         796.54 |          9.05 |          87.99x |       98.86% |
| 128        |          10.69 |          0.35 |          30.23x |       96.69% |
| 256        |          21.11 |          0.39 |          53.58x |       98.13% |
| 500        |        4039.26 |         41.35 |          97.67x |       98.98% |
| 512        |          44.09 |          0.45 |          97.24x |       98.97% |
| 1000       |        8015.93 |         82.01 |          97.74x |       98.98% |
| 1024       |          88.35 |          0.72 |         122.39x |       99.18% |

**Summary Statistics:**
- **Average Speedup**: 59.48x
- **Maximum Speedup**: 162.54x
- **Minimum Speedup**: 1.79x

## Scalability Analysis

## Phase 2 Performance Targets

### Micro-Benchmark Targets

Target: 20% improvement across all single operations

| Benchmark | Baseline (μs) | Target (μs) |
|-----------|---------------|-------------|
| test_encode_small_data_16                |        0.3102 |      0.2482 |
| test_encode_medium_data_64               |        0.3326 |      0.2661 |
| test_encode_single_value                 |        0.3631 |      0.2905 |
| test_encode_tiny_data_4                  |        0.3044 |      0.2435 |
| test_encode_large_data_256               |        0.3678 |      0.2942 |
| ... and 6 more | | |

### Comparative Performance Targets

Target: 1.5x improvement in speedup factor OR minimum 50x speedup

| Data Size | Current Speedup | Target Speedup | Improvement Needed |
|-----------|-----------------|----------------|-------------------|
| null       |          162.54x |         243.81x | 50.0%             |
| 1          |           17.09x |          50.00x | 192.5%            |
| 4          |            1.79x |          50.00x | 2685.5%           |
| 8          |            2.79x |          50.00x | 1693.0%           |
| 10         |           66.64x |          99.96x | 50.0%             |
| 16         |            5.01x |          50.00x | 897.4%            |
| 32         |            8.75x |          50.00x | 471.7%            |
| 50         |           85.21x |         127.82x | 50.0%             |
| 64         |           15.04x |          50.00x | 232.4%            |
| 100        |           87.99x |         131.99x | 50.0%             |
| 128        |           30.23x |          50.00x | 65.4%             |
| 256        |           53.58x |          80.37x | 50.0%             |
| 500        |           97.67x |         146.51x | 50.0%             |
| 512        |           97.24x |         145.86x | 50.0%             |
| 1000       |           97.74x |         146.61x | 50.0%             |
| 1024       |          122.39x |         183.58x | 50.0%             |

## Detailed Benchmark Results

### Batch Encode Dimensions

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_dim_8                            |   46.9704 |   9.2500 | 209.9580 |     23.2271 |     42.1250 |   7735 |
| test_encode_batch_dim_16                           |   46.8197 |   9.7910 | 2129.1670 |     35.1290 |     41.1670 |   6955 |
| test_encode_batch_dim_32                           |   48.2180 |   9.7500 | 212.9170 |     22.4691 |     42.8750 |   9157 |
| test_encode_batch_dim_64                           |   57.6329 |  13.0000 | 1718.2080 |     32.6317 |     50.7080 |   8929 |
| test_encode_batch_dim_128                          |   60.6845 |  20.1670 | 278.3330 |     26.8905 |     53.2500 |   4185 |
| test_encode_batch_dim_256                          |   84.7735 |  24.0830 | 520.5830 |     37.7661 |     78.0420 |   6043 |
| test_encode_batch_dim_512                          |  106.3676 |  35.0420 | 1874.8330 |     49.9591 |     98.8750 |   4411 |

### Batch Encode Edge

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_dim_smaller_than_qubits           |   38.3752 |   7.2500 | 221.4170 |     21.3155 |     33.3330 |   9231 |
| test_encode_batch_dim_larger_than_qubits           |   38.7666 |   5.6670 | 1738.7910 |     26.8808 |     33.3340 |  11691 |
| test_encode_batch_zeros                            |   40.9739 |   7.0840 | 1297.2500 |     28.3418 |     35.2920 |   9285 |
| test_encode_batch_ones                             |   35.5189 |   7.0840 | 172.3330 |     18.4716 |     30.6670 |   7173 |

### Batch Encode Mixed

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_small_batch_large_dim            |   32.2325 |   8.1670 | 174.1250 |     18.4987 |     27.0830 |   6180 |
| test_encode_batch_large_batch_small_dim            |   90.9452 |  25.1660 | 1683.9580 |     42.9016 |     80.5625 |   6306 |
| test_encode_batch_balanced                         |   69.0132 |  14.5000 | 254.0000 |     31.8950 |     62.5830 |   5527 |

### Batch Encode Qubits

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_qubits_4                         |   41.2323 |   8.1660 | 198.5000 |     19.1712 |     36.7705 |  10192 |
| test_encode_batch_qubits_8                         |   39.1207 |   8.5000 | 218.2500 |     14.5091 |     36.5000 |  11263 |
| test_encode_batch_qubits_16                        |   47.2073 |   8.3330 | 2000.2920 |     30.2783 |     41.5830 |  13202 |
| test_encode_batch_qubits_32                        |   49.8027 |  10.3330 | 256.3750 |     23.4719 |     45.3330 |   5963 |
| test_encode_batch_qubits_64                        |   63.4255 |  13.0420 | 297.5000 |     30.2694 |     56.9170 |   9841 |

### Batch Encode Scaling

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_scaling[10-16]         [10-16]   |   19.1818 |   3.7910 | 1951.7910 |     22.0412 |     15.9580 |  12566 |
| test_encode_batch_scaling[50-32]         [50-32]   |   39.8471 |   8.5420 | 220.4590 |     21.7994 |     34.4580 |   6050 |
| test_encode_batch_scaling[100-64]        [100-64]  |   63.2253 |  13.5000 | 2275.2500 |     41.6747 |     55.8750 |   7895 |
| test_encode_batch_scaling[500-128]       [500-128] |  130.8999 |  49.5000 | 2871.2080 |     67.6888 |    124.5000 |   5309 |
| test_encode_batch_scaling[1000-256]      [1000-256] |  445.5526 | 292.9170 | 2598.7920 |    109.7149 |    432.5420 |   1929 |
| test_encode_batch_scaling[5000-256]      [5000-256] | 1568.6631 | 1323.5830 | 5039.5410 |    220.5358 |   1539.0830 |    487 |

### Batch Encode Single

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_single_row                       |    0.4524 |   0.3330 |   5.5000 |      0.0660 |      0.4580 |  39409 |
| test_encode_batch_small_8x4                        |    0.5346 |   0.4666 |   9.4583 |      0.0577 |      0.5250 | 172652 |
| test_encode_batch_medium_32x16                     |   31.9688 |   7.0000 | 148.1250 |     17.5151 |     27.5410 |   1445 |
| test_encode_batch_large_128x32                     |   50.0955 |  10.4590 | 2062.1250 |     33.1712 |     44.1670 |   7586 |

### Batch Encode Size

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_size_1                           |    0.3923 |   0.3438 |   7.4341 |      0.0506 |      0.3854 | 191976 |
| test_encode_batch_size_10                          |   24.6798 |   5.8340 | 154.0000 |     14.7720 |     19.0830 |   6199 |
| test_encode_batch_size_50                          |   49.0035 |   9.0830 | 1736.0830 |     30.5027 |     43.8750 |  10413 |
| test_encode_batch_size_100                         |   53.3824 |  13.4580 | 377.7920 |     23.9037 |     47.0840 |   7742 |
| test_encode_batch_size_500                         |   90.7373 |  41.5410 | 678.7500 |     35.4862 |     81.7500 |   4373 |
| test_encode_batch_size_1000                        |  134.8773 |  68.3750 | 1606.0000 |     53.6470 |    127.7500 |   2717 |
| test_encode_batch_size_5000                        |  588.5208 | 430.3750 | 3323.1250 |    139.1144 |    571.7915 |   1424 |
| test_encode_batch_size_10000                       | 1164.2421 | 826.0000 | 4164.5830 |    197.2730 |   1142.0835 |    744 |

### Batch Encode Throughput

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_batch_throughput_small                 |   50.3377 |   9.2920 | 2519.3330 |     39.9061 |     44.1670 |   6826 |
| test_encode_batch_throughput_medium                |  114.7871 |  36.5420 | 1795.8330 |     57.5539 |    105.2500 |   5395 |
| test_encode_batch_throughput_large                 | 1077.7343 | 763.0000 | 4116.9580 |    265.2482 |   1034.8335 |    826 |

### Comparative Batch

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_numpy_baseline_batch_small                    |   21.3077 |  20.2080 |  51.9580 |      1.1532 |     20.9590 |  34935 |
| test_simd_batch_small                              |    0.7115 |   0.6429 |  13.5119 |      0.0621 |      0.7024 | 191976 |
| test_numpy_baseline_batch_medium                   |  791.4315 | 775.1250 | 910.0830 |     13.7422 |    787.8750 |   1198 |
| test_simd_batch_medium                             |    8.6950 |   7.8750 |  24.2500 |      0.4685 |      8.6250 |  77173 |
| test_numpy_baseline_batch_large                    | 15897.5033 | 15718.3330 | 16411.1670 |    140.1171 |  15877.3340 |     63 |
| test_simd_batch_large                              |   92.4718 |  89.5840 | 114.7920 |      2.7085 |     91.4170 |   6067 |

### Comparative Scaling

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_numpy_encode_scaling[4]             [4]       |    0.5934 |   0.5416 |   4.0709 |      0.0468 |      0.5854 |  81914 |
| test_numpy_encode_scaling[8]             [8]       |    0.9187 |   0.8854 |   3.6250 |      0.0475 |      0.9063 |  53692 |
| test_numpy_encode_scaling[16]            [16]      |    1.5868 |   1.4375 |  21.7290 |      0.1225 |      1.5625 | 157904 |
| test_numpy_encode_scaling[32]            [32]      |    2.9252 |   2.7910 |  49.6250 |      0.2867 |      2.8750 | 196734 |
| test_numpy_encode_scaling[64]            [64]      |    5.5642 |   5.1250 |  22.7500 |      0.3605 |      5.5000 | 136370 |
| test_numpy_encode_scaling[128]           [128]     |   10.6927 |  10.4160 |  23.8750 |      0.5745 |     10.5830 |  67605 |
| test_numpy_encode_scaling[256]           [256]     |   21.1056 |  19.2920 | 206.7500 |      1.5839 |     20.8330 |  42253 |
| test_numpy_encode_scaling[512]           [512]     |   44.0903 |  41.5420 |  70.7090 |      1.9036 |     43.4580 |  20220 |
| test_numpy_encode_scaling[1024]          [1024]    |   88.3503 |  86.7090 | 112.1250 |      2.7180 |     87.3750 |  11066 |
| test_simd_encode_scaling[4]              [4]       |    0.3306 |   0.3006 |   1.0595 |      0.0215 |      0.3274 | 195123 |
| test_simd_encode_scaling[8]              [8]       |    0.3294 |   0.2945 |   1.2667 |      0.0212 |      0.3277 | 193573 |
| test_simd_encode_scaling[16]             [16]      |    0.3165 |   0.2861 |   5.1223 |      0.0340 |      0.3139 | 192012 |
| test_simd_encode_scaling[32]             [32]      |    0.3344 |   0.3041 |   1.8937 |      0.0228 |      0.3312 | 142026 |
| test_simd_encode_scaling[64]             [64]      |    0.3699 |   0.3334 |   1.6667 |      0.0250 |      0.3654 | 198334 |
| test_simd_encode_scaling[128]            [128]     |    0.3537 |   0.3223 |  23.0611 |      0.0588 |      0.3500 | 186047 |
| test_simd_encode_scaling[256]            [256]     |    0.3939 |   0.3590 |   1.3141 |      0.0248 |      0.3910 | 180474 |
| test_simd_encode_scaling[512]            [512]     |    0.4534 |   0.4125 |  11.0334 |      0.0547 |      0.4500 | 188965 |
| test_simd_encode_scaling[1024]           [1024]    |    0.7219 |   0.5333 |   1.9250 |      0.0774 |      0.7250 | 104801 |
| test_numpy_batch_scaling[1]              [1]       |    8.4354 |   8.1250 |  18.9170 |      0.4863 |      8.3330 |  71429 |
| test_numpy_batch_scaling[10]             [10]      |   80.2935 |  73.4590 | 156.5000 |      2.9672 |     79.2910 |  12043 |
| ... and 10 more | | | | | | |

### Comparative Single

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_numpy_baseline_small                          |    1.7137 |   1.5830 |  24.0420 |      0.2170 |      1.7080 |  64000 |
| test_simd_small                                    |    0.3398 |   0.2945 |  27.0222 |      0.0689 |      0.3361 | 190476 |
| test_numpy_baseline_medium                         |    5.6081 |   5.4160 |  19.4160 |      0.3726 |      5.5420 |  88559 |
| test_simd_medium                                   |    0.3627 |   0.3334 |   0.9333 |      0.0233 |      0.3583 | 131148 |
| test_numpy_baseline_large                          |   21.2541 |  20.7920 |  52.3330 |      1.0131 |     20.9580 |  39604 |
| test_simd_large                                    |    0.4031 |   0.3680 |   1.4931 |      0.0273 |      0.3993 | 188965 |

### Comparative Speedup

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_speedup_small_data_64                         |    0.4003 |   0.3558 |   4.6763 |      0.0301 |      0.3974 | 200000 |
| test_speedup_medium_data_1024                      |    0.6778 |   0.5209 |   3.0396 |      0.0693 |      0.6750 |  56469 |
| test_speedup_large_data_8192                       |    6.1669 |   5.6250 |  16.3340 |      0.4056 |      6.0840 |  89888 |
| test_speedup_batch_small                           |    0.7019 |   0.6500 |   7.8604 |      0.0573 |      0.6938 |  73395 |
| test_speedup_batch_medium                          |    8.9810 |   8.2500 |  23.3330 |      0.5040 |      8.8750 |  77173 |
| test_speedup_batch_large                           |   92.4805 |  88.9170 | 119.3330 |      3.0069 |     91.2920 |   4512 |

### Encode Edge

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_data_smaller_than_qubits               |    0.2956 |   0.2616 |   4.1319 |      0.0268 |      0.2939 | 190477 |
| test_encode_data_smaller_than_qubits               |    0.2962 |   0.2745 |  10.4975 |      0.0379 |      0.2916 | 195123 |
| test_encode_data_larger_than_qubits                |    0.2986 |   0.2721 |   0.9632 |      0.0198 |      0.2965 | 198373 |
| test_encode_zeros                                  |    0.3455 |   0.3145 |   1.1979 |      0.0212 |      0.3416 | 146349 |
| test_encode_ones                                   |    0.3440 |   0.3214 |   1.2946 |      0.0230 |      0.3393 | 193536 |
| test_encode_mixed_values                           |    0.2970 |   0.2685 |  28.3078 |      0.0676 |      0.2939 | 195123 |

### Encode Qubits

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_4_qubits                               |    0.3044 |   0.2732 |   4.3426 |      0.0214 |      0.3009 | 190476 |
| test_encode_8_qubits                               |    0.3576 |   0.2500 |  10.0410 |      0.0589 |      0.3750 | 195122 |
| test_encode_16_qubits                              |    0.2961 |   0.2672 |   2.2475 |      0.0201 |      0.2917 | 193536 |
| test_encode_32_qubits                              |    0.2974 |   0.2794 |   0.9681 |      0.0196 |      0.2941 | 190476 |
| test_encode_64_qubits                              |    0.3348 |   0.3125 |   0.9166 |      0.0207 |      0.3312 | 142006 |

### Encode Repeated

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_repeated_small                         |    7.2934 |   6.3340 |  46.0830 |      0.4971 |      7.2500 |  58679 |
| test_encode_repeated_medium                        |    8.7795 |   7.7910 |  51.7910 |      0.5746 |      8.6670 |  60151 |
| test_encode_repeated_small                         |    7.4357 |   6.9160 |  18.2090 |      0.4458 |      7.3750 |  53692 |
| test_encode_repeated_medium                        |    8.9420 |   8.1250 | 211.2920 |      1.0439 |      8.8750 |  73622 |
| test_encode_repeated_large                         |   13.4416 |  12.2910 |  65.5000 |      0.8218 |     13.2910 |  54921 |

### Encode Scaling

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_scaling[8]                   [8]       |    0.2993 |   0.2760 |   2.5234 |      0.0216 |      0.2943 | 200000 |
| test_encode_scaling[16]                  [16]      |    0.3024 |   0.2760 |   3.5495 |      0.0234 |      0.2995 | 196734 |
| test_encode_scaling[32]                  [32]      |    0.3083 |   0.2769 |   1.1544 |      0.0184 |      0.3064 | 196696 |
| test_encode_scaling[64]                  [64]      |    0.3501 |   0.3028 |   2.5806 |      0.0245 |      0.3472 | 190476 |
| test_encode_scaling[128]                 [128]     |    0.4004 |   0.2920 |  10.8330 |      0.0810 |      0.3750 | 114824 |
| test_encode_scaling[256]                 [256]     |    0.3706 |   0.3452 |   2.0595 |      0.0268 |      0.3661 | 188965 |
| test_encode_scaling[512]                 [512]     |    0.4926 |   0.4160 |   8.2080 |      0.0969 |      0.5000 |  76924 |
| test_encode_scaling[1024]                [1024]    |    0.6997 |   0.5250 |   4.8250 |      0.0815 |      0.7125 | 107620 |

### Encode Single

| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| test_encode_small_data_16                          |    0.3553 |   0.2500 |   8.1250 |      0.1023 |      0.3340 |  53099 |
| test_encode_medium_data_64                         |    0.3311 |   0.2917 |   1.1406 |      0.0260 |      0.3281 | 189001 |
| test_encode_single_value                           |    0.3631 |   0.2910 |   8.8750 |      0.0723 |      0.3750 | 109088 |
| test_encode_tiny_data_4                            |    0.3044 |   0.2656 |   4.0286 |      0.0291 |      0.3021 | 191976 |
| test_encode_small_data_16                          |    0.3102 |   0.2708 |  52.5651 |      0.1840 |      0.3021 | 198374 |
| test_encode_medium_data_64                         |    0.3326 |   0.3125 |   1.3154 |      0.0224 |      0.3304 | 193536 |
| test_encode_large_data_256                         |    0.3678 |   0.3462 |   1.4680 |      0.0237 |      0.3622 | 193536 |
| test_encode_xlarge_data_1024                       |    0.7079 |   0.5041 |   3.5958 |      0.0720 |      0.7041 | 109589 |
| test_encode_xxlarge_data_4096                      |    3.0321 |   2.3340 | 719.0420 |      2.1814 |      3.0000 | 110096 |

## Recommendations for Phase 2

### Performance Optimization Priorities

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
