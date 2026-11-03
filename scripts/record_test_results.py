"""
Records automated test results to artifacts/reports/test_results.json.
"""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

def record_tests():
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    p = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    
    # Parse test names and status
    lines = p.stdout.splitlines()
    test_cases = []
    passed = 0
    failed = 0
    
    for line in lines:
        if " PASSED" in line:
            test_name = line.split(" PASSED")[0].strip()
            test_cases.append({"test": test_name, "status": "PASSED"})
            passed += 1
        elif " FAILED" in line:
            test_name = line.split(" FAILED")[0].strip()
            test_cases.append({"test": test_name, "status": "FAILED"})
            failed += 1

    report = {
        "test_suite_status": "ALL_TESTS_PASSED" if failed == 0 else "TESTS_FAILED",
        "total_tests": len(test_cases),
        "passed_tests": passed,
        "failed_tests": failed,
        "execution_exit_code": p.returncode,
        "test_modules_covered": [
            "tests/test_drift.py",
            "tests/test_edge_cases_and_robustness.py",
            "tests/test_features.py",
            "tests/test_ingestion.py",
            "tests/test_leakage_adversarial.py",
            "tests/test_models.py",
            "tests/test_onnx_serving.py",
            "tests/test_promotion_governance.py",
            "tests/test_retraining.py",
            "tests/test_validation.py"
        ],
        "test_cases": test_cases,
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat()
    }

    out_file = ROOT_DIR / "artifacts/reports/test_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Recorded test results to {out_file}: {passed} passed, {failed} failed.")

if __name__ == "__main__":
    record_tests()
