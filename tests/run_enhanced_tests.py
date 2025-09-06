#!/usr/bin/env python3
"""
Enhanced Capture System Test Runner

Comprehensive test runner for all enhanced capture system components.
Provides detailed reporting, coverage analysis, and performance metrics.
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Enhanced test runner with detailed reporting."""
    
    def __init__(self, verbose=False, coverage=True):
        self.verbose = verbose
        self.coverage = coverage
        self.results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
    
    def run_test_suite(self, test_patterns: List[str] = None) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("🧠 Enhanced Second Brain - Test Suite Runner")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Default test patterns if none provided
        if not test_patterns:
            test_patterns = [
                "test_unified_capture_service.py",
                "test_advanced_capture_service.py", 
                "test_enhanced_apple_shortcuts_service.py",
                "test_enhanced_discord_service.py",
                "test_api_integration.py"
            ]
        
        # Run each test module
        for pattern in test_patterns:
            print(f"\n🔍 Running: {pattern}")
            print("-" * 40)
            
            result = self._run_single_test(pattern)
            self.results[pattern] = result
            
            self._update_totals(result)
            self._print_test_summary(pattern, result)
        
        # Generate final report
        self._generate_final_report()
        
        return self.results
    
    def _run_single_test(self, test_file: str) -> Dict[str, Any]:
        """Run a single test file."""
        test_path = project_root / "tests" / test_file
        
        if not test_path.exists():
            return {
                "status": "error",
                "error": f"Test file not found: {test_path}",
                "duration": 0,
                "tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", str(test_path)]
        
        if self.verbose:
            cmd.extend(["-v", "-s"])
        else:
            cmd.append("-q")
        
        if self.coverage:
            cmd.extend([
                "--cov=services",
                "--cov-report=term-missing",
                "--cov-append"
            ])
        
        cmd.extend([
            "--tb=short",
            "--json-report",
            f"--json-report-file={project_root}/tests/.pytest_cache/{test_file}.json"
        ])
        
        # Run test
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300  # 5 minute timeout
            )
            duration = time.time() - start_time
            
            # Parse pytest JSON output if available
            json_report_path = project_root / "tests" / ".pytest_cache" / f"{test_file}.json"
            test_stats = self._parse_pytest_json(json_report_path)
            
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                **test_stats
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "Test execution timed out after 5 minutes",
                "duration": time.time() - start_time,
                "tests": 0,
                "passed": 0, 
                "failed": 0,
                "skipped": 0
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time,
                "tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
    
    def _parse_pytest_json(self, json_path: Path) -> Dict[str, Any]:
        """Parse pytest JSON report for detailed stats."""
        try:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                return {
                    "tests": summary.get('total', 0),
                    "passed": summary.get('passed', 0),
                    "failed": summary.get('failed', 0),
                    "skipped": summary.get('skipped', 0),
                    "errors": summary.get('error', 0)
                }
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not parse JSON report: {e}")
        
        return {
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }
    
    def _update_totals(self, result: Dict[str, Any]):
        """Update total test statistics."""
        self.total_tests += result.get('tests', 0)
        self.passed_tests += result.get('passed', 0)
        self.failed_tests += result.get('failed', 0)
        self.skipped_tests += result.get('skipped', 0)
    
    def _print_test_summary(self, test_file: str, result: Dict[str, Any]):
        """Print summary for a single test file."""
        status = result.get('status', 'unknown')
        duration = result.get('duration', 0)
        
        status_emoji = {
            'passed': '✅',
            'failed': '❌',
            'error': '💥',
            'timeout': '⏰'
        }.get(status, '❓')
        
        print(f"{status_emoji} {test_file}: {status.upper()} ({duration:.2f}s)")
        
        if status == 'passed':
            tests = result.get('tests', 0)
            passed = result.get('passed', 0)
            failed = result.get('failed', 0)
            skipped = result.get('skipped', 0)
            
            print(f"   📊 {tests} total, {passed} passed, {failed} failed, {skipped} skipped")
        
        elif status in ['failed', 'error', 'timeout']:
            error_msg = result.get('error', result.get('stderr', 'Unknown error'))
            if error_msg and self.verbose:
                print(f"   💬 {error_msg[:200]}...")
    
    def _generate_final_report(self):
        """Generate comprehensive final test report."""
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 60)
        print("🏁 FINAL TEST REPORT")
        print("=" * 60)
        
        # Overall statistics
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        print(f"📊 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"⏭️  Skipped: {self.skipped_tests}")
        
        # Success rate
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Test suite breakdown
        print("\n📋 Test Suite Breakdown:")
        print("-" * 30)
        
        for test_file, result in self.results.items():
            status = result.get('status', 'unknown')
            duration = result.get('duration', 0)
            tests = result.get('tests', 0)
            
            status_symbol = {
                'passed': '✅',
                'failed': '❌',
                'error': '💥',
                'timeout': '⏰'
            }.get(status, '❓')
            
            print(f"{status_symbol} {test_file:<35} | {tests:>3} tests | {duration:>6.2f}s")
        
        # Performance insights
        self._print_performance_insights()
        
        # Coverage summary (if enabled)
        if self.coverage:
            self._print_coverage_summary()
        
        # Recommendations
        self._print_recommendations()
    
    def _print_performance_insights(self):
        """Print performance insights."""
        print("\n⚡ Performance Insights:")
        print("-" * 25)
        
        # Find slowest and fastest tests
        test_times = [(name, result.get('duration', 0)) for name, result in self.results.items()]
        test_times.sort(key=lambda x: x[1], reverse=True)
        
        if test_times:
            slowest = test_times[0]
            fastest = test_times[-1]
            
            print(f"🐌 Slowest: {slowest[0]} ({slowest[1]:.2f}s)")
            print(f"🚀 Fastest: {fastest[0]} ({fastest[1]:.2f}s)")
        
        # Test efficiency
        total_time = sum(result.get('duration', 0) for result in self.results.values())
        avg_time_per_test = total_time / max(1, self.total_tests)
        print(f"📊 Average per test: {avg_time_per_test:.3f}s")
    
    def _print_coverage_summary(self):
        """Print test coverage summary."""
        print("\n🛡️  Test Coverage:")
        print("-" * 18)
        
        # Try to read coverage report
        coverage_file = project_root / ".coverage"
        if coverage_file.exists():
            print("✅ Coverage report generated")
            print("📄 Run 'coverage report' for detailed analysis")
        else:
            print("⚠️  No coverage data found")
    
    def _print_recommendations(self):
        """Print test recommendations."""
        print("\n💡 Recommendations:")
        print("-" * 20)
        
        # Analyze results and provide recommendations
        failed_suites = [name for name, result in self.results.items() if result.get('status') != 'passed']
        
        if not failed_suites:
            print("🎉 All tests passing! Consider adding more edge cases.")
        else:
            print(f"🔧 {len(failed_suites)} test suite(s) need attention:")
            for suite in failed_suites:
                print(f"   • {suite}")
        
        # Performance recommendations
        slow_tests = [(name, result.get('duration', 0)) for name, result in self.results.items() 
                     if result.get('duration', 0) > 10]
        
        if slow_tests:
            print(f"⚡ Consider optimizing {len(slow_tests)} slow test(s)")
        
        # Coverage recommendations
        if self.coverage:
            print("📊 Review coverage report and add tests for uncovered code")
    
    def run_specific_tests(self, test_names: List[str]):
        """Run specific test functions or classes."""
        print(f"🎯 Running specific tests: {', '.join(test_names)}")
        
        cmd = [sys.executable, "-m", "pytest"] + test_names
        if self.verbose:
            cmd.extend(["-v", "-s"])
        
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
    
    def run_performance_tests(self):
        """Run performance-focused tests."""
        print("⚡ Running performance tests...")
        
        performance_markers = [
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-sort=mean"
        ]
        
        cmd = [sys.executable, "-m", "pytest", "tests/"] + performance_markers
        
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
    
    def run_integration_tests_only(self):
        """Run only integration tests."""
        print("🔗 Running integration tests only...")
        
        return self.run_test_suite(["test_api_integration.py"])


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Enhanced Capture System Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_enhanced_tests.py                    # Run all tests
  python run_enhanced_tests.py --verbose          # Verbose output
  python run_enhanced_tests.py --no-coverage      # Skip coverage
  python run_enhanced_tests.py --integration-only # Only integration tests
  python run_enhanced_tests.py --performance      # Performance tests only
  python run_enhanced_tests.py --specific test_unified_capture_service.py::test_text_capture_success
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true", 
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run only performance tests"
    )
    
    parser.add_argument(
        "--specific",
        nargs="+",
        help="Run specific test functions or classes"
    )
    
    parser.add_argument(
        "--tests",
        nargs="+",
        help="Run specific test files"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(
        verbose=args.verbose,
        coverage=not args.no_coverage
    )
    
    # Determine what to run
    if args.specific:
        success = runner.run_specific_tests(args.specific)
        sys.exit(0 if success else 1)
    
    elif args.performance:
        success = runner.run_performance_tests()
        sys.exit(0 if success else 1)
    
    elif args.integration_only:
        results = runner.run_integration_tests_only()
        
    else:
        # Run full test suite
        test_patterns = args.tests if args.tests else None
        results = runner.run_test_suite(test_patterns)
    
    # Exit with appropriate code
    failed_count = sum(1 for result in results.values() if result.get('status') != 'passed')
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()