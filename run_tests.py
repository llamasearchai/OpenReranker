#!/usr/bin/env python3
"""
Comprehensive test runner for OpenReranker.

This script provides various testing options including unit tests,
integration tests, coverage reports, and linting.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        return False
    else:
        print(f"✅ Command succeeded: {cmd}")
        return True


def setup_environment():
    """Set up the test environment."""
    print("Setting up test environment...")
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Install the package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return False
    
    # Install test dependencies
    test_deps = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "pytest-mock>=3.10.0",
        "pytest-asyncio>=0.21.0",
        "httpx>=0.24.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "coverage[toml]>=7.0.0"
    ]
    
    for dep in test_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    return True


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = "pytest tests/"
    
    if verbose:
        cmd += " -v"
    
    if coverage:
        cmd += " --cov=open_reranker --cov-report=html --cov-report=term-missing"
    
    return run_command(cmd, "Running unit tests")


def run_linting():
    """Run linting checks."""
    success = True
    
    # Black formatting check
    if not run_command("black --check --diff open_reranker tests", "Checking code formatting with Black"):
        success = False
    
    # isort import sorting check
    if not run_command("isort --check-only --diff open_reranker tests", "Checking import sorting with isort"):
        success = False
    
    # Flake8 linting
    if not run_command("flake8 open_reranker tests", "Running flake8 linting"):
        success = False
    
    return success


def run_type_checking():
    """Run type checking with mypy."""
    return run_command("mypy open_reranker", "Running type checking with mypy")


def format_code():
    """Format code with black and isort."""
    success = True
    
    if not run_command("black open_reranker tests", "Formatting code with Black"):
        success = False
    
    if not run_command("isort open_reranker tests", "Sorting imports with isort"):
        success = False
    
    return success


def run_api_tests():
    """Run API integration tests."""
    # Start the server in the background and run API tests
    cmd = """
    uvicorn open_reranker.main:app --host 127.0.0.1 --port 8001 &
    SERVER_PID=$!
    sleep 5
    pytest tests/test_api.py -v
    TEST_RESULT=$?
    kill $SERVER_PID 2>/dev/null || true
    exit $TEST_RESULT
    """
    return run_command(cmd, "Running API integration tests")


def run_security_checks():
    """Run security checks."""
    success = True
    
    # Install security tools if not available
    run_command("pip install bandit safety", "Installing security tools")
    
    # Run bandit security linter
    if not run_command("bandit -r open_reranker", "Running security checks with bandit"):
        success = False
    
    # Run safety check for known vulnerabilities
    if not run_command("safety check", "Checking for known vulnerabilities with safety"):
        success = False
    
    return success


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="OpenReranker test runner")
    parser.add_argument("--setup", action="store_true", help="Set up test environment")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--all", action="store_true", help="Run all tests and checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # No arguments provided, run basic tests
        args.unit = True
        args.lint = True
    
    success = True
    
    if args.setup or args.all:
        if not setup_environment():
            success = False
    
    if args.format:
        if not format_code():
            success = False
    
    if args.unit or args.all:
        if not run_unit_tests(verbose=args.verbose, coverage=args.coverage or args.all):
            success = False
    
    if args.api or args.all:
        if not run_api_tests():
            success = False
    
    if args.lint or args.all:
        if not run_linting():
            success = False
    
    if args.type_check or args.all:
        if not run_type_checking():
            success = False
    
    if args.security or args.all:
        if not run_security_checks():
            success = False
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests and checks passed!")
        sys.exit(0)
    else:
        print("❌ Some tests or checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 