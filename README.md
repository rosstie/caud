# Landscape Complexity Shapes the Role of Network Density and Diversity in Collective Adaptation Under Disruption

This repository contains the core model code to accompany the manuscript: *Landscape Complexity Shapes the Role of Network Density and Diversity in Collective Adaptation Under Disruption*.

## Overview

This codebase implements a simulation model that examines how complex adaptive systems respond to disruption. The model uses NK fitness landscapes to represent problem complexity and simulates multi-agent groups that adapt through social learning and innovation within network structures. The model explores how network density, diversity, and landscape complexity interact to shape collective adaptation under various disruption scenarios. The models extends builds on the approach from *Network structure shapes  the impact of diversity in collective  learning* - Baumann, Czaplicka & Rahwan (2024)

## Key Features

- **NK Landscape Modeling**: Configurable NK fitness landscapes (K values 0-7) representing different problem complexities
- **Multi-Agent Simulation**: Groups of agents that learn and adapt over time
- **Network Topologies**: Support for various network types (Erdős–Rényi, Watts-Strogatz, etc.)
- **Disaster Modeling**: Configurable disruption events that modify the fitness landscape
- **Social Learning & Innovation**: Agents adapt through both social learning and individual innovation
- **Comprehensive Metrics**: Tracks performance, diversity, learning events, and recovery metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd caud
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation by running the example script:
   ```bash
   python example.py
   ```

This will run a simple simulation and generate a verification plot in `results/fig/example_verification.png`.

## Quick Start

### Running a Simple Verification

The fastest way to verify the model works is to run the example script:

```bash
python example.py
```

This runs a minimal simulation with default parameters and displays key results including:
- Average performance over time
- Social learning and innovation events
- Disaster events (if configured)
- Timing information

### Running Simulations

The main entry point for running simulations is `run.py`. 

**Basic usage (single simulation):**
```bash
python run.py --output-dir data/simulation_results
```

This runs a single simulation with default parameters from `scripts/config/params.py` and saves results to the specified directory.

**Running multiple repetitions:**
```bash
python run.py --output-dir data/simulation_results --repetitions 5
```

**Running parameter sweeps:**
```bash
python run.py --output-dir data/simulation_results --repetitions 5 --use-parameters
```

The `--use-parameters` flag uses the `parameters()` function which defines arrays of parameter values. All combinations will be run.

**Parallel execution:**
```bash
python run.py --output-dir data/simulation_results --repetitions 5 --workers 4
```

### Configuration

Simulation parameters can be configured in `scripts/config/params.py`:

- `get_parameters()`: Returns a single parameter set (used by default)
- `parameters()`: Returns parameter arrays for parameter sweeps (used with `--use-parameters`)

Key parameters include:
- `numberOfAgents`: Number of agents in the simulation
- `numberOfAgentGroups`: Number of agent groups
- `numberOfTimeSteps`: Simulation duration
- `N_NKmodel`, `K_NKmodel`: NK landscape parameters (N=15, K=0-7)
- `typeOfNetworkSocialLearning`: Network type ('er', 'ws', etc.)
- `p_erNetwork`: Edge probability for Erdős–Rényi networks
- `disasterProbability`: Probability of disruption events
- `disasterDistributionType`: Distribution type for disaster impacts

### Test Simulations with Visualization

For quick test runs with visualization, you can use either:

**Option 1: Simple verification (recommended):**
```bash
python example.py
```

**Option 2: Test script with custom parameters:**
```bash
python scripts/run_test.py
```

This runs a simulation using parameters from `scripts/params_test.py` and generates a visualization plot in `results/fig/`. You can modify `scripts/params_test.py` to customize the simulation parameters.

## Project Structure

```
caud/
├── run.py                    # Main simulation runner (primary entry point)
├── example.py                # Quick verification script
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
│
├── scripts/
│   ├── core/                 # Core model components
│   │   ├── simulation.py     # Main simulation class
│   │   ├── landscape.py      # NK fitness landscape
│   │   ├── network.py        # Network topologies
│   │   ├── disaster.py       # Disaster/disruption modeling
│   │   └── metrics.py        # Metrics collection
│   │
│   ├── config/
│   │   └── params.py         # Parameter configuration
│   │
│   ├── utils/                # Utility functions
│   │   ├── measures.py       # Metric calculations
│   │   ├── storage.py        # Results storage (Zarr)
│   │   └── utils.py          # Helper functions
│   │
│   ├── post_processing/      # Advanced post-processing tools (optional)
│   │   ├── convert_zarr_to_parquet.py
│   │   ├── convert_zarr_timeseries_to_parquet.py
│   │   ├── load_raw_data.py
│   │   └── ...
│   │
│   ├── run_test.py           # Test simulations with visualization
│   └── run_post_processing.py # Post-processing script
│
└── tests/                    # Test suite (optional)
    ├── unit/                 # Unit tests
    └── integration/          # Integration tests
```

## Output Format

Simulation results are saved in Zarr format with the following structure:

```
data/simulation_results/
└── {parameter_hash}/
    └── run_{run_id}.zarr/
        ├── meta/              # Simulation metadata
        ├── network/           # Network metrics
        ├── disaster/          # Disaster events and impacts
        ├── summary/           # Summary statistics
        └── recovery/          # Recovery metrics
```

## Post-Processing (Advanced)

Extended post-processing tools are available in `scripts/post_processing/` for:
- Converting Zarr results to Parquet format
- Aggregating results across runs
- Advanced analysis and visualization

See `scripts/post_processing/` for details. These tools are optional and not required for basic model verification.

## Testing

Run the test suite (optional):

```bash
python run_tests.py
```

Or run specific test categories:
```bash
python run_tests.py --unit          # Run unit tests only
python run_tests.py --integration   # Run integration tests only
python run_tests.py --verbose       # Verbose output
```

Or use pytest directly:
```bash
pytest tests/
```

Note: Some tests may fail if post-processing scripts have been moved or if test data is missing. This is expected for a publication repository.

## Citation

If you use this code in your research, please cite the associated publication:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please contact: [Contact information to be added]
