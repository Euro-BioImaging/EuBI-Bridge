#!/usr/bin/env python3
"""
Quick Reference: Running the EuBI-Bridge Test Suite

This script provides examples of how to run the comprehensive test suite
for the `eubi to_zarr` command.
"""

# ============================================================================
# BASIC TEST EXECUTION
# ============================================================================

# Run all tests
#   pytest tests/ -v

# Run tests with minimal output
#   pytest tests/ -q

# Run tests with very verbose output (shows individual assertions)
#   pytest tests/ -vv

# ============================================================================
# RUNNING SPECIFIC TEST MODULES
# ============================================================================

# Unary conversion tests only
#   pytest tests/test_unary_conversions.py -v

# OME-TIFF metadata tests only
#   pytest tests/test_ome_metadata_parsing.py -v

# Aggregative conversion tests only
#   pytest tests/test_aggregative_conversions.py -v

# Parameter interaction tests only
#   pytest tests/test_parameter_interactions.py -v

# ============================================================================
# RUNNING SPECIFIC TEST CLASSES
# ============================================================================

# Test zarr format (v2 vs v3)
#   pytest tests/test_unary_conversions.py::TestZarrFormat -v

# Test chunking configuration
#   pytest tests/test_unary_conversions.py::TestChunking -v

# Test pixel metadata (scales, units)
#   pytest tests/test_unary_conversions.py::TestPixelMetadata -v

# Test OME-TIFF metadata reading
#   pytest tests/test_ome_metadata_parsing.py::TestOMEMetadataReading -v

# Test channel concatenation
#   pytest tests/test_aggregative_conversions.py::TestChannelConcatenationCategorical -v

# ============================================================================
# RUNNING SPECIFIC TEST FUNCTIONS
# ============================================================================

# Single test: zarr v2 format
#   pytest tests/test_unary_conversions.py::TestZarrFormat::test_zarr_format_v2 -v

# Single test: zarr v3 format
#   pytest tests/test_unary_conversions.py::TestZarrFormat::test_zarr_format_v3 -v

# Single test: read OME channel names
#   pytest tests/test_ome_metadata_parsing.py::TestOMEMetadataReading::test_read_ome_channel_names -v

# Single test: Z concatenation
#   pytest tests/test_aggregative_conversions.py::TestZConcatenation::test_z_concat_basic -v

# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================

# Generate coverage report (terminal + HTML)
#   pytest tests/ --cov=eubi_bridge --cov-report=term-missing --cov-report=html

# View coverage in browser (after running above)
#   open htmlcov/index.html

# Coverage report with XML (for CI)
#   pytest tests/ --cov=eubi_bridge --cov-report=xml

# ============================================================================
# DEBUGGING AND TROUBLESHOOTING
# ============================================================================

# Show print statements (normally suppressed by pytest)
#   pytest tests/ -v -s

# Stop on first failure
#   pytest tests/ -x

# Show last N lines of error output
#   pytest tests/ -v --tb=short

# Show full traceback (very detailed)
#   pytest tests/ -v --tb=long

# Run tests with increased verbosity and show locals
#   pytest tests/ -vv -l

# Keep temp files for inspection (modify conftest.py to not auto-cleanup)
#   pytest tests/ -v --basetemp=/tmp/pytest-eubi

# ============================================================================
# FILTERING TESTS BY NAME
# ============================================================================

# Run all tests containing "zarr" in name
#   pytest tests/ -k zarr -v

# Run all tests containing "ome" in name
#   pytest tests/ -k ome -v

# Run all tests containing "channel" in name
#   pytest tests/ -k channel -v

# Run tests NOT containing "aggregative"
#   pytest tests/ -k "not aggregative" -v

# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

# Install pytest-xdist for parallel execution
#   pip install pytest-xdist

# Run tests in parallel (uses all CPU cores)
#   pytest tests/ -n auto -v

# Run tests with specific number of workers
#   pytest tests/ -n 4 -v

# ============================================================================
# CI/CD ENVIRONMENT
# ============================================================================

# Simulate CI environment (offline mode)
#   export MAVEN_OFFLINE=true
#   export JGO_CACHE_DIR=/dev/null
#   pytest tests/ -v

# Run as CI would on Ubuntu
#   docker run -it -v $(pwd):/workspace -w /workspace python:3.12
#   pip install -e .
#   pip install pytest numpy tifffile
#   pytest tests/ -v

# ============================================================================
# TEST SUITE STATISTICS
# ============================================================================

# List all available tests
#   pytest tests/ --collect-only

# Count total tests
#   pytest tests/ --collect-only | grep "test_"

# Show tests matching pattern
#   pytest tests/ --collect-only -q | grep zarr

# ============================================================================
# FIXTURE INSPECTION
# ============================================================================

# List all available fixtures
#   pytest tests/ --fixtures | grep "imagej\|ome\|aggregative"

# Show fixture with description
#   pytest tests/conftest.py --fixtures

# ============================================================================
# ADVANCED PATTERNS
# ============================================================================

# Run specific tests only on macOS
#   pytest tests/ -v -m macos  (requires @pytest.mark.macos)

# Run only fast tests (skip slow ones)
#   pytest tests/ -v -m "not slow"  (requires @pytest.mark.slow)

# Run tests and generate JSON report
#   pytest tests/ --json-report --json-report-file=report.json

# Generate JUnit XML report (for Jenkins/GitLab)
#   pytest tests/ --junit-xml=junit.xml

# ============================================================================
# EXAMPLE WORKFLOWS
# ============================================================================

# Workflow 1: Quick smoke test
#
#   pytest tests/test_unary_conversions.py::TestZarrFormat -v
#   pytest tests/test_aggregative_conversions.py::TestZConcatenation::test_z_concat_basic -v

# Workflow 2: Full test with coverage
#
#   pytest tests/ --cov=eubi_bridge --cov-report=html --cov-report=term-missing
#   open htmlcov/index.html

# Workflow 3: Debug single failing test
#
#   pytest tests/test_unary_conversions.py::TestZarrFormat::test_zarr_format_v3 -vv -s --tb=long

# Workflow 4: Validate OME metadata handling
#
#   pytest tests/test_ome_metadata_parsing.py -v

# Workflow 5: Test aggregative conversions only
#
#   pytest tests/test_aggregative_conversions.py -v

# Workflow 6: Run in CI offline mode
#
#   export MAVEN_OFFLINE=true
#   export JGO_CACHE_DIR=/dev/null
#   pytest tests/ -v --tb=short

# ============================================================================
# USEFUL COMMAND COMBINATIONS
# ============================================================================

# Run tests, show coverage, and generate HTML report
#   pytest tests/ -v --cov=eubi_bridge --cov-report=term-missing --cov-report=html && open htmlcov/index.html

# Run failing tests with full output
#   pytest tests/ --lf -vv -s

# Run tests that failed last, plus new tests
#   pytest tests/ --ff -v

# Run tests and save output to file
#   pytest tests/ -v > test_results.txt 2>&1

# ============================================================================
# INSTALLATION FOR TESTING
# ============================================================================

# Install EuBI-Bridge in development mode
#   pip install -e .

# Install test dependencies
#   pip install pytest pytest-cov numpy tifffile imageio

# Optional: Install for parallel execution
#   pip install pytest-xdist

# Optional: Install for advanced reporting
#   pip install pytest-json-report

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Test fails with "ImportError: No module named 'eubi_bridge'"
#   Solution: Run `pip install -e .` in the project root

# Test fails with "JVM not found" or Bio-Formats errors
#   Solution: Ensure Java is installed: `java -version`
#   On macOS: `brew install openjdk@11`
#   On Ubuntu: `sudo apt-get install openjdk-11-jdk`

# Test fails with "tifffile not found"
#   Solution: `pip install tifffile`

# Tests are slow
#   Solution: Run specific tests instead of full suite
#   Solution: Use pytest-xdist for parallel execution: `pip install pytest-xdist && pytest tests/ -n auto`

# Temporary files not cleaned up
#   Check: /tmp/pytest-* directories
#   These are automatically cleaned by pytest after test completion

# ============================================================================

print(__doc__)
