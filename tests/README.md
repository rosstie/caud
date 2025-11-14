# Test Suite

This directory contains the test suite for the model. The tests are organized into three categories:

## Test Categories

### Unit Tests (`unit/`)
- Tests for individual components
- Focus on testing specific functionality in isolation
- Examples: `test_disaster.py`, `test_landscape.py`

### Integration Tests (`integration/`)
- Tests for interactions between components
- Focus on testing how components work together
- Examples: `test_disaster_landscape.py`

### Visualization Tests (`visualization/`)
- Tests that generate visualizations
- Only run when explicitly requested with `--runvisualization`
- Examples: Visualization tests in `test_disaster.py`

## Test Structure

The project has two test directories:

1. **Root-level `tests/` directory** (recommended for new tests):
   - Contains the organized test structure with unit, integration, and visualization subdirectories
   - Follows standard Python project structure
   - All new tests should be added here

2. **Legacy `scripts/tests/` directory**:
   - Contains older test files that haven't been migrated to the new structure
   - These tests can be run with the `--legacy` flag
   - Will eventually be migrated to the root-level `tests/` directory

## Running Tests

### Basic Usage
```bash
# Run all tests in the new structure
python run_tests.py

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --visualization

# Run legacy tests
python run_tests.py --legacy

# Run with visualization
python run_tests.py --visualization --runvisualization

# Run with coverage report
python run_tests.py --coverage
python run_tests.py --coverage --html  # Generate HTML coverage report
```

### Using pytest Directly
```bash
# Run all tests in the new structure
pytest

# Run specific test categories
pytest tests/unit
pytest tests/integration
pytest tests/visualization

# Run legacy tests
pytest scripts/tests

# Run with visualization
pytest --runvisualization

# Run with coverage
pytest --cov=scripts --cov-report=term
pytest --cov=scripts --cov-report=html
```

## Test Organization

The test suite is organized as follows:

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── test_disaster.py   # Tests for the disaster model
│   ├── test_landscape.py  # Tests for the fitness landscape
│   └── ...
├── integration/           # Integration tests
│   ├── test_disaster_landscape.py  # Tests for disaster-landscape interaction
│   └── ...
├── visualization/         # Visualization tests
│   └── ...
├── conftest.py           # pytest configuration
└── README.md             # This file

scripts/
└── tests/                # Legacy tests (to be migrated)
    ├── test_save_load.py
    ├── test_post_processing.py
    └── ...
```

## Writing Tests

### Unit Tests
- Focus on testing a single component
- Use fixtures to set up test data
- Use assertions to verify expected behavior

### Integration Tests
- Test interactions between components
- Use fixtures to set up multiple components
- Verify that components work together correctly

### Visualization Tests
- Mark with `@pytest.mark.visualization`
- Only run when `--runvisualization` is specified
- Save visualizations to a temporary directory

## Test Fixtures

Common fixtures are defined in the test files:

- `disaster_params`: Parameters for the disaster model
- `disaster_model`: An initialized disaster model
- `landscape_params`: Parameters for the fitness landscape
- `basic_landscape`: A K=0 landscape
- `complex_landscape`: A K=7 landscape
- `integration_params`: Parameters for integration tests

## Logging

Tests use the Python logging module to provide information about test execution:

- Log files are stored in the `logs/` directory
- Log level is set to INFO by default
- Log format includes timestamp, level, and message