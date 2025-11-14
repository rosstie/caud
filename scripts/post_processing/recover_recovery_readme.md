# Recovery Metrics Recalculation Utility

This script (recover_recovery_metrics.py) recalculates and saves recovery metrics for simulation runs that may be missing or need updating. It processes simulation results stored in Zarr format and writes the recovery metrics back to each run directory.
Usage
`python scripts/recover_recovery_metrics.py --input <simulation_data_dir>` 

`--input` (required): Path to the directory containing simulation run Zarr files (e.g., data/simulation).

## What It Does

Scans all run directories (e.g., run_*.zarr) within the specified input directory.

For each run, loads the time series data (payoff_mean, impact_rmsd).

Recalculates recovery metrics using the latest logic in scripts/utils/measures.py.

Saves the recovery metrics in a flat structure to a recovery subdirectory within each run.

## Requirements

Python 3.8+
Dependencies: xarray, numpy, os, argparse
Ensure the updated scripts/utils/measures.py is present (contains the latest calculate_recovery_metrics).

## Example

`python scripts/recover_recovery_metrics.py --input data/simulation --data/recovered_recovery`

This will process all runs in data/simulation, updating or creating the recovery dataset for each.

## Notes

The script is idempotent: you can safely rerun it to update recovery metrics after code changes.
If a run is missing required time series data, a warning will be logged and an empty recovery dataset will be created for that run.