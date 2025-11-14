# RATCASES: Resilience Analysis Tool for Complex Adaptive Systems

## Overview
RATCASES is a Python-based framework for analyzing resilience in complex adaptive systems using NK fitness landscapes. The framework simulates how different system configurations (represented by K0-K7 landscapes) respond to turbulence and disasters, providing insights into system resilience and adaptation strategies.

## Project Structure
```
RATCASES/
├── scripts/                 # Core implementation
│   ├── core/               # Core functionality
│   │   ├── landscape.py    # NK landscape implementation
│   │   ├── network.py      # Network topology and interactions
│   │   ├── disaster.py     # Disaster modeling
│   │   ├── simulation.py   # Main simulation engine
│   │   └── metrics.py      # Performance metrics
│   ├── utils/              # Utility functions
│   ├── config/             # Configuration files
│   └── logs/               # Log files
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── visualization/     # Visualization tests
├── data/                  # Data files
├── results/               # Simulation results
└── docs/                  # Documentation
```

## Key Features
- NK landscape modeling with configurable K values (0-7)
- Multi-agent group simulation
- Disaster impact modeling
- Turbulence-based landscape changes
- Comprehensive metrics for system performance
- Extensive test coverage

## Testing Approach
The project includes a comprehensive test suite that verifies:

1. **Basic Functionality**
   - Landscape initialization
   - Zero turbulence behavior
   - Non-zero turbulence effects
   - Metric calculations

2. **Multi-Group Behavior**
   - Group-specific turbulence effects
   - Inter-group interactions
   - Group metric calculations

3. **Turbulence Scaling**
   - Non-linear response to turbulence
   - Magnitude of changes
   - Structure preservation

4. **K0 vs K7 Differences**
   - Structural preservation
   - Change magnitude
   - Turbulence scaling

## Installation
1. Clone the repository
2. Create a conda environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate ratcases
   ```

## Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/unit/test_quantization.py

# Run with verbose output
python -m pytest -v
```

## Usage

### Configuring and Running Simulations

1. **Configure simulation parameters**:
   - Edit parameters in `scripts/params_test.py` to set up your simulation
   - Key parameters include:
     - Number of agents and groups
     - Network type (e.g., ER, WS)
     - NK landscape parameters (N, K)
     - Disaster parameters (probability, distribution)

2. **Run test simulations**:
   ```bash
   python scripts/run_test.py
   ```
   This will run simulations using parameters from params_test.py and store results in `data/simulation_test/`.

3. **Run full-scale simulations**:
   ```bash
   python run.py --output-dir data/simulation_results --repetitions 5
   ```
   The simulation results are saved in Zarr format with the following structure:
   ```
   data/simulation_results/{parameter_hash}/run_{run_id}.zarr/
   ```

### Post-Processing and Analysis

1. **Post-process simulation results**:
   ```bash
   python scripts/run_post_processing.py --input data/simulation_test --output data/processed_test
   ```
   This aggregates results across multiple runs and saves them as Parquet files:
   ```
   data/processed_test/{parameter_hash}/
   ├── time_series.parquet  # Time-series data
   ├── summary.parquet      # Summary statistics
   ├── disaster.parquet     # Disaster metrics
   └── network.parquet      # Network metrics
   ```

2. **Using parallel processing**:
   ```bash
   # Default - use parallel processing with Dask
   python scripts/run_post_processing.py --input data/simulation_test --output data/processed_test
   
   # Sequential processing (for debugging)
   python scripts/run_post_processing.py --input data/simulation_test --output data/processed_test --no-parallel
   
   # Specify number of workers and memory
   python scripts/run_post_processing.py --input data/simulation_test --output data/processed_test --num-workers 4 --memory-per-worker 2
   ```

3. **Analyzing Results**:
   The processed Parquet files can be analyzed using pandas or Dask:
   ```python
   import pandas as pd
   import dask.dataframe as dd
   
   # Load data with pandas
   df = pd.read_parquet('data/processed_test/{parameter_hash}/summary.parquet')
   
   # Load data with Dask for larger datasets
   ddf = dd.read_parquet('data/processed_test/*/summary.parquet')
   
   # Filter by parameter values
   filtered = ddf[ddf.param_numberOfAgentGroups == 2]
   
   # Perform analysis
   result = filtered.groupby('param_disasterProbability').mean().compute()
   ```

## Dependencies
- Python 3.12+
- NumPy
- SciPy
- Matplotlib
- PyBDM
- pytest
- xarray
- zarr
- pandas
- dask
- pyarrow

## Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License
[Your License Here]

## Contact
[Your Contact Information]


## Running on a VM
### Prerequisites

Ensure your VM has the following installed:
- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - networkx
  - xarray
  - dask
  - zarr
  - pyarrow (for Parquet support)

### Running a Test Simulation

1. Navigate to the scripts directory:
   ```
   cd scripts
   ```

2. Run a test simulation:
   ```
   python run_test.py
   ```
   
   This will run a simulation with the default parameters and generate a plot in the `../results/fig/` directory.

3. Modify the parameters for run_test.py in `params_test.py` to customize your test simulation.

### Running full-scale simulations:
 
You can run large simulations by changing params.py in scripts, putting values as arrays and single values in () while using the `--use-parameters` instruction will run all the combinations of parameters defined in params.py. 

as an example the following code will run 5 simulations for all parameter combinations saving to: 

```bash
python run.py --output-dir data/simulation_results --repetitions 5 --use-parameters
```
The simulation results are saved in Zarr format with the following structure:
```
data/simulation_results/{parameter_hash}/run_{run_id}.zarr/

### Post-Processing Results
#### Aggegrating data accross runs for parameter values:

If you've run multiple simulations and saved results in Zarr format, you can aggregate them:

```
python run_post_processing.py --input /path/to/results --output /path/to/output
```

Options:
- `--no-parallel`: Disable parallel processing
- `--num-workers N`: Set number of workers (default: automatic)
- `--memory-per-worker X`: Set memory per worker in GB (default: automatic)
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

#### Converting to parquet but while keeing runs separate
Statistical analysis of results often requires standalone values for each run on parameter set. 

Combined raw results can be obtained by using the following command: 

Parquet files of non aggregated data can be created using convert_zarr_to_parquet.py, specific variables can be selected using the following 

specific strings: 
```bash
python scripts/convert_zarr_to_parquet.py --input /mnt/data/sim_results_recovery --output /mnt/data/processed_results_raw --selected-vars "run_id,meta.K_NK,meta.number_of_groups,meta.number_of_timesteps,meta.N_NK,meta.p_er_network,network.density,network.mean_degree,disaster.count,disaster.average_impact_magnitude,disaster.average_impact_rmsd,disaster.total_impact_magnitude,disaster.total_impact_rmsd,summary.average_payoff_mean,summary.average_payoff_median,summary.cumulative_payoff_mean,summary.cumulative_payoff_median,summary.cumulative_payoff_variability,summary.average_item_diversity,summary.variability_item_diversity,summary.cumulative_unique_items,recovery.avg_drop_pct,recovery.avg_recovery_pct,recovery.avg_time_to_recovery,recovery.full_recovery_rate,recovery.max_recovery_time,recovery.threshold,recovery.window,summary.per_timestep_social_learning,summary.per_timestep_innovation,summary.cumulative_social_learning,summary.cumulative_innovation"
```

```bash
python scripts/convert_zarr_to_parquet.py --input /mnt/data/simulation_50_results --output /mnt/data/processed_50_results_raw --selected-vars "run_id,meta.K_NK,meta.number_of_groups,meta.number_of_timesteps,meta.N_NK,meta.p_er_network,network.density,network.mean_degree,disaster.count,disaster.average_impact_magnitude,disaster.average_impact_rmsd,disaster.total_impact_magnitude,disaster.total_impact_rmsd,summary.average_payoff_mean,summary.average_payoff_median,summary.cumulative_payoff_mean,summary.cumulative_payoff_median,summary.cumulative_payoff_variability,summary.average_item_diversity,summary.variability_item_diversity,summary.cumulative_unique_items,recovery.avg_drop_pct,recovery.avg_recovery_pct,recovery.avg_time_to_recovery,recovery.full_recovery_rate,recovery.max_recovery_time,recovery.threshold,recovery.window,summary.per_timestep_social_learning,summary.per_timestep_innovation,summary.cumulative_social_learning,summary.cumulative_innovation"
```
python scripts/convert_zarr_to_parquet.py --input /mnt/data/simulation_50_results_control --output /mnt/data/processed_50_results_control_raw --selected-vars "run_id,meta.K_NK,meta.number_of_groups,meta.number_of_timesteps,meta.N_NK,meta.p_er_network,network.density,network.mean_degree,disaster.count,disaster.average_impact_magnitude,disaster.average_impact_rmsd,disaster.total_impact_magnitude,disaster.total_impact_rmsd,summary.average_payoff_mean,summary.average_payoff_median,summary.cumulative_payoff_mean,summary.cumulative_payoff_median,summary.cumulative_payoff_variability,summary.average_item_diversity,summary.variability_item_diversity,summary.cumulative_unique_items,recovery.avg_drop_pct,recovery.avg_recovery_pct,recovery.avg_time_to_recovery,recovery.full_recovery_rate,recovery.max_recovery_time,recovery.threshold,recovery.window,summary.per_timestep_social_learning,summary.per_timestep_innovation,summary.cumulative_social_learning,summary.cumulative_innovation"


For local 
python scripts/convert_zarr_to_parquet.py --input data/simulation_50_results_control --output data/processed_50_results_control_raw --selected-vars "run_id,meta.K_NK,meta.number_of_groups,meta.number_of_timesteps,meta.N_NK,meta.p_er_network,network.density,network.mean_degree,disaster.count,disaster.average_impact_magnitude,disaster.average_impact_rmsd,disaster.total_impact_magnitude,disaster.total_impact_rmsd,summary.average_payoff_mean,summary.average_payoff_median,summary.cumulative_payoff_mean,summary.cumulative_payoff_median,summary.cumulative_payoff_variability,summary.average_item_diversity,summary.variability_item_diversity,summary.cumulative_unique_items,recovery.avg_drop_pct,recovery.avg_recovery_pct,recovery.avg_time_to_recovery,recovery.full_recovery_rate,recovery.max_recovery_time,recovery.threshold,recovery.window,summary.per_timestep_social_learning,summary.per_timestep_innovation,summary.cumulative_social_learning,summary.cumulative_innovation"

## Visualization

Simulation results are automatically visualized and saved to the `../results/fig/` directory.

## Performance Optimization

This codebase has been optimized for VM deployment:
- Debug print statements have been removed or replaced with proper logging
- Memory usage has been optimized for better performance
- Network initialization has been improved for better connectivity
- Parallelization options are available for post-processing

## Contact

For questions or support, please contact: [your-email@example.com]
