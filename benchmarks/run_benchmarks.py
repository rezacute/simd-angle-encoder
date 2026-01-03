#!/usr/bin/env python3
"""
Convenience script to run all benchmarks and generate reports.

Usage:
    python benchmarks/run_benchmarks.py              # Run all benchmarks
    python benchmarks/run_benchmarks.py --encode     # Run encode benchmarks only
    python benchmarks/run_benchmarks.py --batch      # Run batch benchmarks only
    python benchmarks/run_benchmarks.py --compare    # Run comparative benchmarks only
    python benchmarks/run_benchmarks.py --scale      # Run scalability benchmarks only
    python benchmarks/run_benchmarks.py --report     # Generate summary report
    python benchmarks/run_benchmarks.py --fast       # Run quick subset (for development)
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class BenchmarkRunner:
    """Manage benchmark execution and report generation."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent / "reports"
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and report results."""
        print(f"\n{'='*70}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print('='*70)

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"❌ FAILED: {description}")
            return False

        print(f"✅ SUCCESS: {description}")
        return True

    def run_encode_benchmarks(self, fast: bool = False):
        """Run single encoding benchmarks."""
        cmd = [
            "pytest", "benchmarks/python/test_bench_encode.py",
            "--benchmark-only",
            "--benchmark-autosave",
            "-v"
        ]

        if fast:
            cmd.extend(["-k", "small or medium"])

        success = self.run_command(cmd, "Single Encoding Benchmarks")
        self.results['encode'] = success

    def run_batch_benchmarks(self, fast: bool = False):
        """Run batch encoding benchmarks."""
        cmd = [
            "pytest", "benchmarks/python/test_bench_encode_batch.py",
            "--benchmark-only",
            "--benchmark-autosave",
            "-v"
        ]

        if fast:
            cmd.extend(["-k", "size_1 or size_10 or size_50"])

        success = self.run_command(cmd, "Batch Encoding Benchmarks")
        self.results['batch'] = success

    def run_comparative_benchmarks(self, fast: bool = False):
        """Run comparative benchmarks."""
        cmd = [
            "pytest", "benchmarks/python/test_bench_comparative.py",
            "--benchmark-only",
            "--benchmark-autosave",
            "-v"
        ]

        if fast:
            cmd.extend(["-k", "small"])

        success = self.run_command(cmd, "Comparative Benchmarks")
        self.results['comparative'] = success

    def run_scalability_benchmarks(self, fast: bool = False):
        """Run scalability benchmarks."""
        cmd = [
            "pytest", "benchmarks/python/test_bench_scalability.py",
            "--benchmark-only",
            "--benchmark-autosave",
            "-v"
        ]

        if fast:
            cmd.extend(["-k", "data_size_64 or data_size_256"])

        success = self.run_command(cmd, "Scalability Benchmarks")
        self.results['scalability'] = success

    def run_all_benchmarks(self, fast: bool = False):
        """Run all benchmark suites."""
        print(f"\n🚀 Starting Benchmark Suite")
        print(f"Mode: {'Fast (development)' if fast else 'Full (production)'}")
        print(f"Output directory: {self.output_dir}")

        self.run_encode_benchmarks(fast=fast)
        self.run_batch_benchmarks(fast=fast)
        self.run_comparative_benchmarks(fast=fast)
        self.run_scalability_benchmarks(fast=fast)

    def generate_report(self):
        """Generate summary report from benchmark results."""
        print(f"\n{'='*70}")
        print("Generating Summary Report")
        print('='*70)

        # Find all JSON result files
        json_files = list(self.output_dir.glob("*.json"))

        if not json_files:
            print("⚠️  No benchmark results found to generate report")
            return

        report = {
            "timestamp": datetime.now().isoformat(),
            "suites": {}
        }

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                suite_name = json_file.stem.replace("-results", "")
                report["suites"][suite_name] = self._extract_summary(data)

            except Exception as e:
                print(f"⚠️  Could not process {json_file}: {e}")

        # Write summary report
        report_file = self.output_dir / "summary-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print human-readable summary
        self._print_summary(report)

        # Save markdown report
        md_file = self.output_dir / "summary-report.md"
        self._write_markdown_report(report, md_file)

        print(f"\n✅ Reports generated:")
        print(f"  - JSON: {report_file}")
        print(f"  - Markdown: {md_file}")

    def _extract_summary(self, data: Dict) -> Dict:
        """Extract summary information from benchmark data."""
        summary = {
            "machine_info": data.get("machine_info", {}),
            "benchmarks": []
        }

        for bench in data.get("benchmarks", []):
            summary["benchmarks"].append({
                "name": bench["name"],
                "group": bench.get("group", "unknown"),
                "stats": {
                    "mean_ms": bench["stats"]["mean"] * 1000,
                    "min_ms": bench["stats"]["min"] * 1000,
                    "max_ms": bench["stats"]["max"] * 1000,
                    "stddev_ms": bench["stats"]["stddev"] * 1000,
                    "rounds": bench["stats"]["rounds"],
                }
            })

        return summary

    def _print_summary(self, report: Dict):
        """Print human-readable summary to console."""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print('='*70)

        for suite_name, suite_data in report["suites"].items():
            print(f"\n📊 {suite_name.upper()}")
            print('-' * 70)

            benchmarks = suite_data.get("benchmarks", [])
            if not benchmarks:
                print("  No benchmarks found")
                continue

            # Group by test group
            groups = {}
            for bench in benchmarks:
                group = bench.get("group", "ungrouped")
                if group not in groups:
                    groups[group] = []
                groups[group].append(bench)

            for group_name, group_benchmarks in groups.items():
                print(f"\n  {group_name}:")
                for bench in group_benchmarks[:5]:  # Show first 5 per group
                    name = bench["name"].split("::")[-1]
                    mean = bench["stats"]["mean_ms"]
                    stddev = bench["stats"]["stddev_ms"]
                    print(f"    {name:50s} {mean:8.4f} ± {stddev:6.4f} ms")

                if len(group_benchmarks) > 5:
                    print(f"    ... and {len(group_benchmarks) - 5} more")

    def _write_markdown_report(self, report: Dict, output_file: Path):
        """Write markdown report file."""
        with open(output_file, 'w') as f:
            f.write("# SIMD Angle Encoder - Benchmark Report\n\n")
            f.write(f"**Generated**: {report['timestamp']}\n\n")

            for suite_name, suite_data in report["suites"].items():
                f.write(f"## {suite_name.title()} Benchmarks\n\n")

                benchmarks = suite_data.get("benchmarks", [])
                if not benchmarks:
                    f.write("No benchmarks found.\n\n")
                    continue

                # Group by test group
                groups = {}
                for bench in benchmarks:
                    group = bench.get("group", "ungrouped")
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(bench)

                for group_name, group_benchmarks in sorted(groups.items()):
                    f.write(f"### {group_name}\n\n")
                    f.write("| Benchmark | Mean (ms) | Min (ms) | Max (ms) | StdDev (ms) | Rounds |\n")
                    f.write("|-----------|-----------|----------|----------|-------------|--------|\n")

                    for bench in group_benchmarks:
                        name = bench["name"].split("::")[-1]
                        stats = bench["stats"]
                        f.write(f"| {name} | {stats['mean_ms']:.4f} | {stats['min_ms']:.4f} | "
                               f"{stats['max_ms']:.4f} | {stats['stddev_ms']:.4f} | {stats['rounds']} |\n")

                    f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run SIMD Angle Encoder benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--encode", action="store_true",
                       help="Run single encoding benchmarks only")
    parser.add_argument("--batch", action="store_true",
                       help="Run batch encoding benchmarks only")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparative benchmarks only")
    parser.add_argument("--scale", action="store_true",
                       help="Run scalability benchmarks only")
    parser.add_argument("--report", action="store_true",
                       help="Generate summary report from existing results")
    parser.add_argument("--fast", action="store_true",
                       help="Run fast subset of benchmarks (for development)")
    parser.add_argument("--output", type=Path,
                       help="Output directory for reports",
                       default=None)

    args = parser.parse_args()

    # If no specific benchmarks selected, run all
    run_all = not (args.encode or args.batch or args.compare or args.scale or args.report)

    runner = BenchmarkRunner(output_dir=args.output)

    if args.report:
        runner.generate_report()
        return 0

    if run_all or args.encode:
        runner.run_encode_benchmarks(fast=args.fast)

    if run_all or args.batch:
        runner.run_batch_benchmarks(fast=args.fast)

    if run_all or args.compare:
        runner.run_comparative_benchmarks(fast=args.fast)

    if run_all or args.scale:
        runner.run_scalability_benchmarks(fast=args.fast)

    # Always generate report at the end
    runner.generate_report()

    # Check if all benchmarks passed
    if all(runner.results.values()):
        print(f"\n{'='*70}")
        print("✅ ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        print('='*70)
        return 0
    else:
        print(f"\n{'='*70}")
        print("⚠️  SOME BENCHMARKS FAILED")
        print('='*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
