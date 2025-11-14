#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to recover recovery metrics from existing simulation results.
This script loads existing simulation results, calculates recovery metrics
from the stored time series data, and saves the updated results.
"""

import os
import sys
import logging
import numpy as np
import xarray as xr
from pathlib import Path
import argparse
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.utils.measures import calculate_recovery_metrics
from scripts.utils.storage import ResultsStorage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recover recovery metrics from existing simulation results"
    )
    parser.add_argument(
        "--input", type=str, help="Path to simulation results directory", required=True
    )
    parser.add_argument(
        "--output", type=str, help="Path to output directory", required=True
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    return parser.parse_args()


def process_run(run_dir, output_dir):
    """Process a single run directory."""
    try:
        # Check if the run directory exists
        if not run_dir.exists():
            logging.warning(f"Run directory {run_dir} does not exist")
            return False

        # Check if the run directory is a directory
        if not run_dir.is_dir():
            logging.warning(f"{run_dir} is not a directory")
            return False

        # Open the Zarr dataset
        logging.info(f"Opening Zarr dataset: {run_dir}")

        # Try to open the time_series dataset
        time_series_path = run_dir / "time_series"
        if not time_series_path.exists():
            logging.warning(f"Time series directory not found: {time_series_path}")
            return False

        logging.info(f"Opening time series dataset: {time_series_path}")

        # Open the dataset
        ds = xr.open_zarr(time_series_path)
        logging.info(f"Loaded dataset: {ds}")

        # Check if required data exists
        if not all(var in ds for var in ["payoff_mean", "impact_rmsd"]):
            logging.warning(f"Required time series data not found in {run_dir}")
            return False

        logging.info(f"Required time series data found in {run_dir}")

        # Extract payoff and impact time series
        payoff_series = ds["payoff_mean"].values
        impact_series = ds["impact_rmsd"].values

        logging.info(f"Payoff series shape: {payoff_series.shape}")
        logging.info(f"Impact series shape: {impact_series.shape}")

        # Calculate recovery metrics
        recovery_metrics = calculate_recovery_metrics(payoff_series, impact_series)

        # Log the keys in recovery_metrics
        logging.info(f"Recovery metrics keys: {list(recovery_metrics.keys())}")

        # Log the values in recovery_metrics
        for key, value in recovery_metrics.items():
            logging.info(f"Recovery metric {key}: {value}")

        # Create output run directory
        output_run_dir = output_dir / run_dir.name
        output_run_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output run directory: {output_run_dir}")

        # Create a new recovery dataset with the calculated metrics
        # Add default values for missing keys
        recovery_data = {}

        # Add metrics directly to the dataset (flat structure)
        for key, value in recovery_metrics.items():
            recovery_data[key] = ((), value)

        recovery_ds = xr.Dataset(
            recovery_data,
            attrs={"description": "Recovery metrics calculated from time series data"},
        )

        logging.info(f"Saving recovery dataset: {recovery_ds}")

        # Save the recovery dataset
        recovery_path = output_run_dir / "recovery"
        recovery_path.mkdir(parents=True, exist_ok=True)
        recovery_ds.to_zarr(str(recovery_path), mode="w")
        logging.info(f"Saved recovery dataset: {recovery_path}")

        # Copy other datasets
        for dataset_name in ["time_series", "summary", "disaster", "network", "meta"]:
            dataset_path = run_dir / dataset_name
            if dataset_path.exists():
                try:
                    # Create the output directory
                    output_dataset_path = output_run_dir / dataset_name
                    output_dataset_path.mkdir(parents=True, exist_ok=True)

                    # Copy the dataset directly
                    ds_copy = xr.open_zarr(dataset_path)
                    ds_copy.to_zarr(str(output_dataset_path), mode="w")
                    logging.info(f"Saved dataset: {output_dataset_path}")
                except Exception as e:
                    logging.warning(f"Error copying dataset {dataset_name}: {e}")

        return True
    except Exception as e:
        logging.error(f"Error processing {run_dir}: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Ensure input and output directories exist
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logging.error(f"Input directory {input_dir} does not exist")
        return 1

    # Find all parameter directories
    param_dirs = [d for d in input_dir.glob("*") if d.is_dir()]

    if not param_dirs:
        logging.error(f"No parameter directories found in {input_dir}")
        return 1

    logging.info(f"Found {len(param_dirs)} parameter directories to process")
    logging.info(f"Parameter directories: {param_dirs}")

    # Process parameter directories
    success_count = 0
    for param_dir in param_dirs:
        logging.info(f"Processing parameter directory: {param_dir}")

        # Find all run directories - look for directories ending with .zarr
        run_dirs = [d for d in param_dir.glob("run_*.zarr") if d.is_dir()]
        if not run_dirs:
            logging.warning(f"No run directories found in {param_dir}")
            continue

        # Create output parameter directory
        output_param_dir = output_dir / param_dir.name
        output_param_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output parameter directory: {output_param_dir}")

        # Process each run
        param_success = 0
        for run_dir in run_dirs:
            logging.info(f"Processing run: {run_dir}")
            if process_run(run_dir, output_param_dir):
                param_success += 1
                logging.info(f"Successfully processed run: {run_dir}")
            else:
                logging.warning(f"Failed to process run: {run_dir}")

        logging.info(
            f"Processed {param_success}/{len(run_dirs)} runs in {param_dir.name}"
        )

        if param_success > 0:
            success_count += 1

    # Report results
    logging.info(
        f"Processed {success_count}/{len(param_dirs)} parameter directories successfully"
    )

    return 0 if success_count == len(param_dirs) else 1


if __name__ == "__main__":
    sys.exit(main())
