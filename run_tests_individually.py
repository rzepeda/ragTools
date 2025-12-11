#!/usr/bin/env python3
"""
Script to run each test individually and save results.
This helps identify failing or hanging tests.
"""
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Configuration
TIMEOUT = 60  # seconds per test
RESULTS_FILE = "individual_test_results.json"
LOG_FILE = "individual_test_results.log"

def get_all_tests():
    """Collect all test names."""
    result = subprocess.run(
        "source venv/bin/activate && pytest --collect-only -q",
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/mnt/MCPProyects/ragTools"
    )
    
    tests = []
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line and '::' in line and not line.startswith('=') and not line.startswith('-'):
            # Remove any trailing info like [  0%]
            test_name = line.split('[')[0].strip()
            if test_name and not test_name.endswith('.py'):
                tests.append(test_name)
    
    return tests

def run_single_test(test_name, index, total):
    """Run a single test and return the result."""
    print(f"[{index}/{total}] Running: {test_name}")
    
    start_time = time.time()
    result = {
        "test": test_name,
        "index": index,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        proc = subprocess.run(
            f"source venv/bin/activate && pytest '{test_name}' -v --tb=short",
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd="/mnt/MCPProyects/ragTools"
        )
        
        duration = time.time() - start_time
        result["duration"] = duration
        result["exit_code"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        
        # Determine status
        if proc.returncode == 0:
            if "PASSED" in proc.stdout:
                result["status"] = "PASSED"
            elif "SKIPPED" in proc.stdout:
                result["status"] = "SKIPPED"
            else:
                result["status"] = "UNKNOWN"
        else:
            result["status"] = "FAILED"
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        result["duration"] = duration
        result["status"] = "TIMEOUT"
        result["exit_code"] = -1
        result["error"] = f"Test timed out after {TIMEOUT} seconds"
        print(f"  ⚠️  TIMEOUT after {TIMEOUT}s")
        
    except Exception as e:
        duration = time.time() - start_time
        result["duration"] = duration
        result["status"] = "ERROR"
        result["exit_code"] = -1
        result["error"] = str(e)
        print(f"  ❌ ERROR: {e}")
    
    # Print status
    status_emoji = {
        "PASSED": "✅",
        "FAILED": "❌",
        "SKIPPED": "⏭️",
        "TIMEOUT": "⏱️",
        "ERROR": "💥",
        "UNKNOWN": "❓"
    }
    emoji = status_emoji.get(result["status"], "❓")
    print(f"  {emoji} {result['status']} ({duration:.2f}s)")
    
    return result

def main():
    """Main execution function."""
    print("=" * 80)
    print("Individual Test Runner")
    print("=" * 80)
    print()
    
    # Get all tests
    print("Collecting tests...")
    tests = get_all_tests()
    total = len(tests)
    print(f"Found {total} tests\n")
    
    # Run each test
    results = []
    stats = {
        "PASSED": 0,
        "FAILED": 0,
        "SKIPPED": 0,
        "TIMEOUT": 0,
        "ERROR": 0,
        "UNKNOWN": 0
    }
    
    start_time = time.time()
    
    for i, test in enumerate(tests, 1):
        result = run_single_test(test, i, total)
        results.append(result)
        stats[result["status"]] += 1
        
        # Save results incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump({
                "total_tests": total,
                "completed": i,
                "stats": stats,
                "results": results
            }, f, indent=2)
    
    total_duration = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print()
    for status, count in stats.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {status:10s}: {count:4d} ({percentage:5.1f}%)")
    
    # Save summary to log file
    with open(LOG_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tests: {total}\n")
        f.write(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)\n\n")
        
        for status, count in stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            f.write(f"{status:10s}: {count:4d} ({percentage:5.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FAILED TESTS\n")
        f.write("=" * 80 + "\n")
        for result in results:
            if result["status"] == "FAILED":
                f.write(f"\n{result['test']}\n")
                if "error" in result:
                    f.write(f"  Error: {result['error']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TIMEOUT TESTS\n")
        f.write("=" * 80 + "\n")
        for result in results:
            if result["status"] == "TIMEOUT":
                f.write(f"\n{result['test']}\n")
                f.write(f"  Duration: {result['duration']:.2f}s\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SKIPPED TESTS\n")
        f.write("=" * 80 + "\n")
        for result in results:
            if result["status"] == "SKIPPED":
                f.write(f"{result['test']}\n")
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Summary saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
