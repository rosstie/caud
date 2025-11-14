#!/usr/bin/env python3
import os
import sys
import argparse
import pytest
from pathlib import Path


def main():
    """Run tests using pytest with various options"""
    parser = argparse.ArgumentParser(description="Run tests for the model")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--visualization", action="store_true", help="Run visualization tests"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--html", action="store_true", help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--test-dir", type=str, default="tests", help="Directory containing tests"
    )
    parser.add_argument(
        "--legacy", action="store_true", help="Run legacy tests from scripts/tests"
    )
    args = parser.parse_args()

    # Default to running all tests if no specific category is selected
    if not (
        args.unit or args.integration or args.visualization or args.all or args.legacy
    ):
        args.all = True

    # Build pytest arguments
    pytest_args = []

    # Add test directories based on selected categories
    test_dir = Path(args.test_dir)
    if args.unit or args.all:
        pytest_args.append(str(test_dir / "unit"))
    if args.integration or args.all:
        pytest_args.append(str(test_dir / "integration"))
    if args.visualization or args.all:
        pytest_args.append(str(test_dir / "visualization"))
    if args.legacy:
        pytest_args.append("scripts/tests")

    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")

    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(["--cov=scripts", "--cov-report=term"])
        if args.html:
            pytest_args.extend(["--cov-report=html"])

    # Run pytest
    result = pytest.main(pytest_args)

    # Return appropriate exit code
    sys.exit(result)


if __name__ == "__main__":
    main()
