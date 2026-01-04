#!/usr/bin/env python3
"""
Comprehensive baseline analysis for SIMD Angle Encoder.

Analyzes all benchmark results and generates:
1. Baseline JSON report with all metrics
2. Comprehensive markdown report with tables and analysis
3. Speedup factors and comparative analysis
4. Phase 2 performance targets
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict
import statistics


class BaselineAnalyzer:
    """Analyze benchmark results and generate comprehensive baseline report."""

    def __init__(self, benchmark_dir: Path):
        self.benchmark_dir = Path(benchmark_dir)
        self.results = []
        self.machine_info = None
        self.commit_info = None

    def load_results(self, json_files: List[Path] = None):
        """Load all benchmark JSON files."""
        if json_files is None:
            # Load the latest benchmark files (files 4-7 from our run)
            json_files = sorted(self.benchmark_dir.glob("*.json"))[-4:]

        print(f"Loading {len(json_files)} benchmark files...")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

                if self.machine_info is None:
                    self.machine_info = data['machine_info']
                    self.commit_info = data['commit_info']

                self.results.extend(data['benchmarks'])

        print(f"Loaded {len(self.results)} total benchmarks")
        return self

    def group_benchmarks(self) -> Dict[str, List[Dict]]:
        """Group benchmarks by their group name."""
        groups = defaultdict(list)
        for bench in self.results:
            group = bench.get('group', 'unknown')
            groups[group].append(bench)
        return dict(groups)

    def calculate_speedup_factors(self) -> Dict[str, Any]:
        """Calculate speedup factors from comparative benchmarks."""
        speedup_data = {}

        # Find comparative benchmarks
        comparative = [b for b in self.results if 'comparative' in b.get('group', '')]
        groups = defaultdict(list)
        for bench in comparative:
            # Extract data size from param if available
            param = bench.get('param', '')
            groups[param].append(bench)

        for param, benches in groups.items():
            simd_times = []
            numpy_times = []

            for bench in benches:
                name = bench['name']
                mean_us = bench['stats']['mean'] * 1_000_000  # Convert to microseconds

                if 'simd' in name.lower():
                    simd_times.append(mean_us)
                elif 'numpy' in name.lower():
                    numpy_times.append(mean_us)

            if simd_times and numpy_times:
                avg_simd = statistics.mean(simd_times)
                avg_numpy = statistics.mean(numpy_times)
                speedup = avg_numpy / avg_simd if avg_simd > 0 else 0

                speedup_data[param] = {
                    'simd_mean_us': avg_simd,
                    'numpy_mean_us': avg_numpy,
                    'speedup_factor': speedup,
                    'improvement_percent': ((1 - avg_simd/avg_numpy) * 100) if avg_numpy > 0 else 0
                }

        return speedup_data

    def analyze_scaling_behavior(self) -> Dict[str, Any]:
        """Analyze scaling behavior across data sizes."""
        scaling_data = {}

        # Group scalability benchmarks
        scalability = [b for b in self.results if 'scalability' in b.get('group', '')]
        groups = defaultdict(list)
        for bench in scalability:
            groups[bench['group']].append(bench)

        for group, benches in groups.items():
            # Sort by data size (extract from param)
            def extract_size_param(bench):
                """Extract numeric size from parameter."""
                param = bench.get('param', '')
                try:
                    # Handle formats like '10-16', '10', etc.
                    if '-' in param:
                        return int(param.split('-')[0])
                    return int(param)
                except (ValueError, TypeError):
                    return 0

            sorted_benches = sorted(benches, key=extract_size_param)

            sizes = []
            times = []

            for bench in sorted_benches:
                param = bench.get('param', '')
                size_param = extract_size_param(bench)

                if size_param > 0:
                    mean_ms = bench['stats']['mean'] * 1000
                    sizes.append(size_param)
                    times.append(mean_ms)

            if len(sizes) >= 2:
                # Calculate scaling factor
                # Linear scaling: time should be proportional to size
                # Check if actual scaling matches expected linear scaling
                scaling_ratios = []
                for i in range(1, len(sizes)):
                    size_ratio = sizes[i] / sizes[i-1]
                    time_ratio = times[i] / times[i-1]
                    if size_ratio > 0:
                        scaling_ratios.append(time_ratio / size_ratio)

                avg_scaling = statistics.mean(scaling_ratios) if scaling_ratios else 1.0

                scaling_data[group] = {
                    'sizes': sizes,
                    'times_ms': times,
                    'scaling_ratio': avg_scaling,
                    'is_linear': 0.8 <= avg_scaling <= 1.2,
                    'throughput_ops_per_ms': [s/t for s, t in zip(sizes, times)]
                }

        return scaling_data

    def calculate_performance_targets(self) -> Dict[str, Any]:
        """Define Phase 2 performance targets based on baseline."""
        targets = {
            'micro_benchmarks': {},
            'macro_benchmarks': {},
            'comparative_targets': {},
            'scalability_targets': {}
        }

        # Micro-benchmark targets (single encode operations)
        encode_benchmarks = [b for b in self.results if 'encode-single' in b.get('group', '')]
        for bench in encode_benchmarks:
            name = bench['name']
            mean_ms = bench['stats']['mean'] * 1000
            baseline_us = mean_ms * 1000

            # Target: 20% improvement for Phase 2
            targets['micro_benchmarks'][name] = {
                'baseline_us': baseline_us,
                'target_us': baseline_us * 0.8,
                'improvement_target': '20%'
            }

        # Comparative targets (speedup factors)
        speedup_factors = self.calculate_speedup_factors()
        for size, data in speedup_factors.items():
            current_speedup = data['speedup_factor']

            # Target: Increase speedup by 1.5x or achieve minimum 50x speedup
            target_speedup = max(current_speedup * 1.5, 50)

            targets['comparative_targets'][size] = {
                'current_speedup': current_speedup,
                'target_speedup': target_speedup,
                'improvement_needed': f"{((target_speedup/current_speedup - 1) * 100):.1f}%"
            }

        # Scalability targets (maintain linear scaling)
        scaling = self.analyze_scaling_behavior()
        for group, data in scaling.items():
            targets['scalability_targets'][group] = {
                'maintain_linear_scaling': data['is_linear'],
                'target_throughput_improvement': '10%'
            }

        return targets

    def generate_baseline_report(self) -> Dict[str, Any]:
        """Generate comprehensive baseline report."""
        print("\nGenerating baseline report...")

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'commit': self.commit_info['id'][:8],
                'branch': self.commit_info['branch'],
                'machine': {
                    'system': self.machine_info['system'],
                    'processor': self.machine_info['processor'],
                    'cpu_brand': self.machine_info['cpu']['brand_raw'],
                    'cpu_count': self.machine_info['cpu']['count'],
                    'python_version': self.machine_info['python_version']
                },
                'total_benchmarks': len(self.results)
            },
            'summary': {
                'total_groups': len(self.group_benchmarks()),
                'total_benchmarks': len(self.results),
                'benchmark_categories': list(self.group_benchmarks().keys())
            },
            'speedup_analysis': self.calculate_speedup_factors(),
            'scaling_analysis': self.analyze_scaling_behavior(),
            'performance_targets': self.calculate_performance_targets(),
            'detailed_results': {}
        }

        # Add detailed results by group
        groups = self.group_benchmarks()
        for group_name, group_benches in groups.items():
            report['detailed_results'][group_name] = []

            for bench in group_benches:
                report['detailed_results'][group_name].append({
                    'name': bench['name'],
                    'param': bench.get('param', ''),
                    'stats': {
                        'mean_us': bench['stats']['mean'] * 1_000_000,
                        'min_us': bench['stats']['min'] * 1_000_000,
                        'max_us': bench['stats']['max'] * 1_000_000,
                        'stddev_us': bench['stats']['stddev'] * 1_000_000,
                        'median_us': bench['stats']['median'] * 1_000_000,
                        'rounds': bench['stats']['rounds']
                    }
                })

        return report

    def save_baseline_json(self, report: Dict[str, Any], output_file: Path):
        """Save baseline report as JSON."""
        print(f"\nSaving baseline JSON to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Saved baseline JSON")

    def generate_markdown_report(self, report: Dict[str, Any], output_file: Path):
        """Generate comprehensive markdown report."""
        print(f"\nGenerating markdown report to {output_file}...")

        with open(output_file, 'w') as f:
            # Title and metadata
            f.write("# SIMD Angle Encoder - Phase 1 Baseline Performance Report\n\n")
            f.write(f"**Generated**: {report['metadata']['timestamp']}\n")
            f.write(f"**Commit**: `{report['metadata']['commit']}`\n")
            f.write(f"**Branch**: {report['metadata']['branch']}\n\n")

            # Machine info
            f.write("## Test Environment\n\n")
            meta = report['metadata']['machine']
            f.write(f"- **System**: {meta['system']} {meta['processor']}\n")
            f.write(f"- **CPU**: {meta['cpu_brand']} ({meta['cpu_count']} cores)\n")
            f.write(f"- **Python**: {meta['python_version']}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"**Total Benchmarks Executed**: {report['metadata']['total_benchmarks']}\n")
            f.write(f"**Benchmark Categories**: {len(report['summary']['benchmark_categories'])}\n\n")

            f.write("### Benchmark Categories\n\n")
            for category in sorted(report['summary']['benchmark_categories']):
                f.write(f"- {category}\n")
            f.write("\n")

            # Speedup Analysis
            f.write("## Performance Analysis: SIMD vs NumPy\n\n")
            speedup = report['speedup_analysis']

            if speedup:
                f.write("| Data Size | NumPy Time (μs) | SIMD Time (μs) | Speedup Factor | Improvement |\n")
                f.write("|-----------|-----------------|---------------|----------------|-------------|\n")

                for size, data in sorted(speedup.items(), key=lambda x: int(x[0]) if x[0] and x[0].isdigit() else 0):
                    size_str = str(size) if size is not None else "null"
                    try:
                        numpy_val = float(data.get('numpy_mean_us', 0.0))
                        simd_val = float(data.get('simd_mean_us', 0.0))
                        speedup_val = float(data.get('speedup_factor', 0.0))
                        improv_val = float(data.get('improvement_percent', 0.0))

                        f.write(f"| {size_str:10s} | {numpy_val:14.2f} | {simd_val:13.2f} | "
                               f"{speedup_val:14.2f}x | {improv_val:11.2f}% |\n")
                    except (TypeError, ValueError) as e:
                        f.write(f"| {size_str:10s} | Error: {str(e)} |\n")

                # Calculate overall speedup
                all_speedups = [d['speedup_factor'] for d in speedup.values()]
                if all_speedups:
                    avg_speedup = statistics.mean(all_speedups)
                    max_speedup = max(all_speedups)
                    min_speedup = min(all_speedups)

                    f.write(f"\n**Summary Statistics:**\n")
                    f.write(f"- **Average Speedup**: {avg_speedup:.2f}x\n")
                    f.write(f"- **Maximum Speedup**: {max_speedup:.2f}x\n")
                    f.write(f"- **Minimum Speedup**: {min_speedup:.2f}x\n\n")
            else:
                f.write("No comparative benchmarks found.\n\n")

            # Scaling Behavior Analysis
            f.write("## Scalability Analysis\n\n")
            scaling = report['scaling_analysis']

            for group, data in sorted(scaling.items()):
                f.write(f"### {group.replace('-', ' ').title()}\n\n")
                f.write(f"**Scaling Ratio**: {data['scaling_ratio']:.2f} (1.0 = perfect linear scaling)\n")
                f.write(f"**Linear Scaling**: {'✅ Yes' if data['is_linear'] else '❌ No'}\n\n")

                f.write("| Data Size | Time (ms) | Throughput (ops/ms) |\n")
                f.write("|-----------|-----------|---------------------|\n")
                for size, time, throughput in zip(data['sizes'], data['times_ms'], data['throughput_ops_per_ms']):
                    f.write(f"| {size:9d} | {time:9.4f} | {throughput:20.2f} |\n")
                f.write("\n")

            # Performance Targets
            f.write("## Phase 2 Performance Targets\n\n")

            # Micro-benchmark targets
            f.write("### Micro-Benchmark Targets\n\n")
            f.write("Target: 20% improvement across all single operations\n\n")

            micro_targets = report['performance_targets']['micro_benchmarks']
            if micro_targets:
                # Show a sample of targets
                sample = list(micro_targets.items())[:5]
                f.write("| Benchmark | Baseline (μs) | Target (μs) |\n")
                f.write("|-----------|---------------|-------------|\n")
                for name, target in sample:
                    f.write(f"| {name:40s} | {target['baseline_us']:13.4f} | {target['target_us']:11.4f} |\n")
                if len(micro_targets) > 5:
                    f.write(f"| ... and {len(micro_targets) - 5} more | | |\n")
                f.write("\n")

            # Comparative targets
            f.write("### Comparative Performance Targets\n\n")
            f.write("Target: 1.5x improvement in speedup factor OR minimum 50x speedup\n\n")

            comp_targets = report['performance_targets']['comparative_targets']
            if comp_targets:
                f.write("| Data Size | Current Speedup | Target Speedup | Improvement Needed |\n")
                f.write("|-----------|-----------------|----------------|-------------------|\n")
                for size, target in sorted(comp_targets.items(), key=lambda x: int(x[0]) if x[0] and x[0].isdigit() else 0):
                    size_str = str(size) if size is not None else "null"
                    f.write(f"| {size_str:10s} | {target['current_speedup']:15.2f}x | {target['target_speedup']:14.2f}x | "
                           f"{target['improvement_needed']:17s} |\n")
                f.write("\n")

            # Detailed Results by Group
            f.write("## Detailed Benchmark Results\n\n")

            for group_name in sorted(report['detailed_results'].keys()):
                f.write(f"### {group_name.replace('-', ' ').title()}\n\n")

                benches = report['detailed_results'][group_name]

                # Create table
                f.write("| Benchmark | Mean (μs) | Min (μs) | Max (μs) | StdDev (μs) | Median (μs) | Rounds |\n")
                f.write("|-----------|-----------|----------|----------|-------------|-------------|--------|\n")

                for bench in benches[:20]:  # Limit to first 20 per group
                    name = bench['name']
                    param = f" [{bench['param']}]" if bench['param'] else ''
                    stats = bench['stats']

                    f.write(f"| {name:40s}{param:10s} | {stats['mean_us']:9.4f} | {stats['min_us']:8.4f} | "
                           f"{stats['max_us']:8.4f} | {stats['stddev_us']:11.4f} | {stats['median_us']:11.4f} | "
                           f"{stats['rounds']:6d} |\n")

                if len(benches) > 20:
                    f.write(f"| ... and {len(benches) - 20} more | | | | | | |\n")

                f.write("\n")

            # Recommendations
            f.write("## Recommendations for Phase 2\n\n")

            f.write("### Performance Optimization Priorities\n\n")

            # Find areas with most opportunity
            if speedup:
                avg_speedup = statistics.mean([d['speedup_factor'] for d in speedup.values()])
                if avg_speedup < 50:
                    f.write("1. **High Priority**: Improve SIMD vectorization to achieve 50x+ speedup\n")
                    f.write(f"   - Current average speedup: {avg_speedup:.2f}x\n")
                    f.write(f"   - Target: 50-90x speedup for batch operations\n")
                    f.write("   - Focus on: Memory alignment, cache optimization, instruction selection\n\n")

            f.write("2. **Medium Priority**: Optimize batch processing throughput\n")
            f.write("   - Implement batch-aware SIMD optimizations\n")
            f.write("   - Reduce per-batch overhead\n")
            f.write("   - Target: 10% improvement in throughput\n\n")

            f.write("3. **Low Priority**: Micro-optimizations for single operations\n")
            f.write("   - Target: 20% improvement\n")
            f.write("   - Focus on: Hot path optimization, branch reduction\n\n")

            f.write("### Testing & Validation\n\n")
            f.write("- Add continuous benchmarking in CI/CD pipeline\n")
            f.write("- Set regression thresholds: 10% warning, 25% critical\n")
            f.write("- Track performance trends across commits\n")
            f.write("- Validate performance on multiple platforms (x86_64, ARM64)\n\n")

            f.write("### Documentation\n\n")
            f.write("- Document performance characteristics in user guide\n")
            f.write("- Create performance tuning guide for different use cases\n")
            f.write("- Add benchmark results to project documentation\n\n")

            f.write("---\n\n")
            f.write("*This report serves as the Phase 1 baseline for all future performance comparisons.*\n")

        print(f"✅ Generated markdown report")


def main():
    """Main entry point."""
    benchmark_dir = Path("/Users/syahriza/data/kubitto/simd-angle-encoder/.benchmarks/Darwin-CPython-3.12-64bit")
    output_dir = Path("/Users/syahriza/data/kubitto/simd-angle-encoder/benchmarks/reports")
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("SIMD Angle Encoder - Baseline Performance Analysis")
    print("="*70)

    # Initialize analyzer
    analyzer = BaselineAnalyzer(benchmark_dir)

    # Load all benchmark results
    analyzer.load_results()

    # Generate comprehensive baseline report
    report = analyzer.generate_baseline_report()

    # Save JSON report
    json_output = output_dir / "baseline.json"
    analyzer.save_baseline_json(report, json_output)

    # Generate markdown report
    md_output = output_dir / "baseline.md"
    analyzer.generate_markdown_report(report, md_output)

    print("\n" + "="*70)
    print("✅ BASELINE ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nReports generated:")
    print(f"  - JSON: {json_output}")
    print(f"  - Markdown: {md_output}")
    print(f"\nTotal benchmarks analyzed: {len(analyzer.results)}")
    print(f"Benchmark groups: {len(analyzer.group_benchmarks())}")

    # Print summary statistics
    speedup = analyzer.calculate_speedup_factors()
    if speedup:
        all_speedups = [d['speedup_factor'] for d in speedup.values()]
        print(f"\nSpeedup Analysis:")
        print(f"  - Average speedup: {statistics.mean(all_speedups):.2f}x")
        print(f"  - Max speedup: {max(all_speedups):.2f}x")
        print(f"  - Min speedup: {min(all_speedups):.2f}x")

    return 0


if __name__ == "__main__":
    exit(main())
