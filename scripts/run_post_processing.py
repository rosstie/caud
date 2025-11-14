#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-processing script for simulation results.
This script aggregates raw simulation results stored in Zarr files to Parquet files.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import xarray as xr
import json
import hashlib

# Set up better path handling
sys.path.append(str(Path(__file__).parent.parent))

from scripts.utils.aggregate_results import (
    aggregate_simulation_results,
    save_aggregated_results,
)
from scripts.utils.measures import calculate_recovery_metrics

# Import Dask components
from dask.distributed import Client, LocalCluster


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process simulation results stored in Zarr format"
    )
    parser.add_argument(
        "--input", type=str, help="Path to simulation results directory", required=True
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output directory for Parquet files",
        required=True,
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
        default=False,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for parallel processing (default: auto)",
        default=None,
    )
    parser.add_argument(
        "--memory-per-worker",
        type=float,
        help="Memory per worker in GB (default: auto)",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of parameter directories to process in each batch.",
        default=None,
    )
    parser.add_argument(
        "--calculate-recovery",
        action="store_true",
        help="Calculate recovery metrics from time series data",
        default=False,
    )
    parser.add_argument(
        "--group-params",
        action="store_true",
        help="Group parameter directories with identical parameters before aggregation",
        default=True,
    )
    return parser.parse_args()


def load_run_data(run_dir):
    """Load data from a single run directory."""
    try:
        # Load xarray datasets
        time_series = xr.open_zarr(run_dir / "time_series")
        summary = xr.open_zarr(run_dir / "summary")
        disaster = xr.open_zarr(run_dir / "disaster")
        network = xr.open_zarr(run_dir / "network")
        meta = xr.open_zarr(run_dir / "meta")

        # Check if recovery dataset exists, if not calculate it
        recovery_path = run_dir / "recovery"
        if recovery_path.exists():
            recovery = xr.open_zarr(recovery_path)
        else:
            # Calculate recovery metrics from time series data
            if "payoff_mean" in time_series and "impact_rmsd" in time_series:
                payoff_series = time_series["payoff_mean"].values
                impact_series = time_series["impact_rmsd"].values

                # Calculate recovery metrics
                recovery_metrics = calculate_recovery_metrics(
                    payoff_series, impact_series
                )

                # Create recovery dataset with flat structure
                recovery_data = {}

                # Add metrics directly to the dataset (flat structure)
                for key, value in recovery_metrics.items():
                    recovery_data[key] = ((), value)

                recovery = xr.Dataset(
                    recovery_data,
                    attrs={
                        "description": "Recovery metrics calculated from time series data"
                    },
                )

                # Save the recovery dataset
                recovery_path.mkdir(parents=True, exist_ok=True)
                recovery.to_zarr(str(recovery_path), mode="w")
                logging.info(f"Calculated and saved recovery metrics for {run_dir}")
            else:
                logging.warning(
                    f"Missing required time series data for recovery metrics in {run_dir}"
                )
                # Create empty recovery dataset with flat structure
                recovery = xr.Dataset(
                    {
                        "count": ((), 0),
                        "avg_drop_pct": ((), 0),
                        "avg_recovery_pct": ((), 0),
                        "full_recovery_rate": ((), 0),
                        "interrupted_recovery_rate": ((), 0),
                        "avg_time_to_recovery": ((), 0),
                        "window": ((), 5),
                        "threshold": ((), 0.01),
                        "max_recovery_time": ((), 50),
                    },
                    attrs={"description": "Empty recovery metrics dataset"},
                )

        # Extract metadata into scalar values
        for var in meta.data_vars:
            if var == "parameters":
                meta[var] = meta[var].values.item()
            else:
                meta[var] = meta[var].values.item()

        # Create run data dictionary
        run_data = {
            "time_series": time_series,
            "summary": summary,
            "disaster": disaster,
            "recovery": recovery,
            "network": network,
            "meta": meta,
        }

        return run_data
    except Exception as e:
        logging.error(f"Error loading {run_dir.name}: {e}")
        return None


def extract_parameters(param_dir):
    """Extract parameter values from a single run directory within the parameter directory."""
    try:
        # Find the first run directory
        run_dirs = list(param_dir.glob("run_*.zarr"))
        if not run_dirs:
            logging.error(f"No run directories found in {param_dir}")
            return None

        run_dir = run_dirs[0]

        # Load metadata
        meta = xr.open_zarr(run_dir / "meta")

        # Extract parameters
        if "parameters" in meta:
            param_data = meta["parameters"].values.item()

            # Handle different parameter formats
            if isinstance(param_data, dict):
                params = param_data
            elif isinstance(param_data, str):
                # Try to evaluate string as dict
                import ast

                try:
                    params = ast.literal_eval(param_data)
                except:
                    logging.warning(f"Could not parse parameters string in {param_dir}")
                    return None
            elif hasattr(param_data, "item"):
                try:
                    dict_data = param_data.item()
                    if isinstance(dict_data, dict):
                        params = dict_data
                    else:
                        return None
                except:
                    return None
            else:
                return None

            # Add basic simulation parameters if not present
            if "numberOfAgents" not in params and "number_of_agents" in meta:
                params["numberOfAgents"] = int(meta["number_of_agents"])
            if "numberOfAgentGroups" not in params and "number_of_groups" in meta:
                params["numberOfAgentGroups"] = int(meta["number_of_groups"])

            return params
        else:
            # Fallback to basic parameters
            return {
                "numberOfAgents": int(meta["number_of_agents"]),
                "numberOfAgentGroups": int(meta["number_of_groups"]),
                "simulationVersion": "extracted",
            }
    except Exception as e:
        logging.error(f"Error extracting parameters from {param_dir}: {e}")
        return None


def create_param_hash(params):
    """Create a hash from parameter values to identify identical parameter sets."""
    if not params:
        return None

    # Create sorted representation of parameters for consistent hashing
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def group_param_dirs_by_params(param_dirs):
    """Group parameter directories by their parameter values."""
    grouped_dirs = {}

    for param_dir in param_dirs:
        params = extract_parameters(param_dir)
        if params:
            param_hash = create_param_hash(params)
            if param_hash:
                if param_hash not in grouped_dirs:
                    grouped_dirs[param_hash] = {"params": params, "dirs": []}
                grouped_dirs[param_hash]["dirs"].append(param_dir)

    # Log grouping results
    for param_hash, group_data in grouped_dirs.items():
        dir_names = [d.name for d in group_data["dirs"]]
        if len(dir_names) > 1:
            logging.info(f"Grouped directories with identical parameters: {dir_names}")

    return grouped_dirs


def process_parameter_directory(param_dir, output_dir):
    """Process all runs in a parameter directory."""
    param_hash = param_dir.name
    logging.info(f"Processing parameter directory: {param_hash}")

    # Find all run directories
    run_dirs = list(param_dir.glob("run_*.zarr"))
    if not run_dirs:
        logging.error(f"No run directories found in {param_dir}")
        return False

    logging.info(f"Found {len(run_dirs)} runs in {param_dir}")

    # Load runs
    runs = []
    for run_dir in run_dirs:
        run_data = load_run_data(run_dir)
        if run_data:
            runs.append(run_data)
            logging.info(f"Successfully loaded {run_dir.name}")

    if not runs:
        logging.error("No runs were loaded successfully")
        return False

    # Extract all parameters from the metadata
    # The 'parameters' in meta contains the full parameter dictionary used for the simulation
    if "parameters" in runs[0]["meta"]:
        # Extract all parameters from the metadata
        params = {}
        # Get the full parameters dictionary from the first run's metadata
        try:
            param_data = runs[0]["meta"]["parameters"].values.item()
            # The parameters might be a string representation of a dict, a numpy array, or a dict directly
            if isinstance(param_data, dict):
                params = param_data
                logging.debug(
                    f"Extracted {len(params)} parameters from metadata (dict type)"
                )
            elif isinstance(param_data, str):
                # Try to evaluate the string as a dict
                import ast

                try:
                    params = ast.literal_eval(param_data)
                    logging.debug(
                        f"Extracted {len(params)} parameters from metadata (string representation)"
                    )
                except:
                    logging.warning(
                        f"Could not parse parameters string: {param_data[:50]}..."
                    )
            elif hasattr(param_data, "item"):
                # Try numpy array .item() method which might return a dict
                try:
                    dict_data = param_data.item()
                    if isinstance(dict_data, dict):
                        params = dict_data
                        logging.debug(
                            f"Extracted {len(params)} parameters from metadata (numpy array)"
                        )
                except:
                    logging.warning(
                        f"Could not extract dict from numpy array: {type(param_data)}"
                    )
            else:
                logging.warning(
                    f"Parameters metadata has unsupported type: {type(param_data)}"
                )

            # If we have params data, add basic simulation parameters if not present
            if (
                params
                and "numberOfAgents" not in params
                and "number_of_agents" in runs[0]["meta"]
            ):
                # Fall back to meta fields if needed
                if (
                    "numberOfAgents" not in params
                    and "number_of_agents" in runs[0]["meta"]
                ):
                    params["numberOfAgents"] = int(runs[0]["meta"]["number_of_agents"])
                if (
                    "numberOfAgentGroups" not in params
                    and "number_of_groups" in runs[0]["meta"]
                ):
                    params["numberOfAgentGroups"] = int(
                        runs[0]["meta"]["number_of_groups"]
                    )
        except Exception as e:
            logging.error(f"Error extracting parameters from metadata: {e}")
            # Fallback to basic parameters if extraction fails
            params = {
                "numberOfAgents": int(runs[0]["meta"]["number_of_agents"]),
                "numberOfAgentGroups": int(runs[0]["meta"]["number_of_groups"]),
                "simulationVersion": "extracted",
            }
    else:
        # Fallback to basic parameters if 'parameters' is not in metadata
        logging.warning("No 'parameters' found in metadata, using basic parameters")
        params = {
            "numberOfAgents": int(runs[0]["meta"]["number_of_agents"]),
            "numberOfAgentGroups": int(runs[0]["meta"]["number_of_groups"]),
            "simulationVersion": "extracted",
        }

    # Add essential information regardless of parameter extraction
    params["dir"] = param_hash  # Add the directory hash
    params["num_runs"] = len(runs)  # Add the number of runs used for aggregation

    logging.debug(f"Parameters for aggregation: {params}")

    # Create output directory
    output_param_dir = output_dir / param_hash

    # Skip if already processed (optional - remove this check if you want to reprocess)
    if output_param_dir.exists() and list(output_param_dir.glob("*.parquet")):
        logging.info(f"Parameter directory {param_hash} already processed, skipping")
        return True

    # Create output directory
    output_param_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate and save results
    try:
        logging.info(f"Aggregating results for {param_hash}...")
        aggregated_data = aggregate_simulation_results(runs, params, len(runs))

        # Debug output to check final structure
        for data_type, df in aggregated_data.items():
            param_cols = [col for col in df.columns if col.startswith("param_")]
            logging.debug(f"{data_type} shape: {df.shape}, param columns: {param_cols}")
            if param_cols:
                logging.debug(
                    f"Sample param values: {df[param_cols].iloc[0].to_dict()}"
                )

        save_aggregated_results(aggregated_data, output_param_dir)
        logging.info(f"Saved aggregated results to {output_param_dir}")
        return True
    except Exception as e:
        logging.error(f"Error during aggregation: {e}")
        # Add more debug information
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        return False


def process_grouped_parameter_directories(group_data, output_dir):
    """Process a group of parameter directories with identical parameters."""
    param_dirs = group_data["dirs"]
    params = group_data["params"]

    # Use the first directory's name for the output
    first_dir_name = param_dirs[0].name
    group_hash = create_param_hash(params)

    logging.info(
        f"Processing parameter group: {group_hash} with {len(param_dirs)} directories"
    )
    logging.info(f"Directories in group: {[d.name for d in param_dirs]}")

    # Create output directory name from group hash
    output_param_dir = output_dir / f"grouped_{group_hash[:8]}"

    # Skip if already processed
    if output_param_dir.exists() and list(output_param_dir.glob("*.parquet")):
        logging.info(f"Parameter group {group_hash[:8]} already processed, skipping")
        return True

    # Collect all runs from all parameter directories in this group
    all_runs = []
    total_run_count = 0

    for param_dir in param_dirs:
        run_dirs = list(param_dir.glob("run_*.zarr"))
        total_run_count += len(run_dirs)

        for run_dir in run_dirs:
            run_data = load_run_data(run_dir)
            if run_data:
                all_runs.append(run_data)
                logging.info(f"Successfully loaded {param_dir.name}/{run_dir.name}")

    if not all_runs:
        logging.error("No runs were loaded successfully from any directory in group")
        return False

    logging.info(
        f"Loaded {len(all_runs)}/{total_run_count} runs from {len(param_dirs)} directories"
    )

    # Add essential information for aggregation
    params["dir"] = group_hash[:8]  # Use shortened group hash
    params["num_runs"] = len(all_runs)  # Total number of runs
    params["grouped_dirs"] = [d.name for d in param_dirs]  # Store original directories

    # Create output directory
    output_param_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate and save results
    try:
        logging.info(f"Aggregating results for parameter group {group_hash[:8]}...")
        aggregated_data = aggregate_simulation_results(all_runs, params, len(all_runs))

        save_aggregated_results(aggregated_data, output_param_dir)
        logging.info(f"Saved aggregated results to {output_param_dir}")

        # Create a mapping file to document which directories were grouped
        mapping_file = output_param_dir / "grouped_directories.json"
        with open(mapping_file, "w") as f:
            json.dump(
                {
                    "group_hash": group_hash,
                    "directories": [d.name for d in param_dirs],
                    "parameter_count": len(params),
                    "run_count": len(all_runs),
                },
                f,
                indent=2,
            )

        return True
    except Exception as e:
        logging.error(f"Error during aggregation of grouped directories: {e}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")
        return False


def process_with_dask(
    param_dirs,
    output_dir,
    num_workers=None,
    memory_per_worker_gb=None,
    batch_size=None,
    group_params=True,
):
    """Process parameter directories in parallel using Dask."""
    from dask.distributed import Client, LocalCluster, wait, as_completed
    import gc  # Import garbage collector here for the main process

    # Determine number of workers based on available resources
    if num_workers is None:
        import multiprocessing

        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Use a reasonable default for memory per worker
    if memory_per_worker_gb is None:
        import psutil

        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        memory_per_worker_gb = max(1, int(total_memory_gb / (num_workers * 2)))
    # Setup cluster with adequate resources
    memory_limit = f"{memory_per_worker_gb}GB"

    # Create a LocalCluster with the specified resources
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,  # Single-threaded workers to avoid contention
        memory_limit=memory_limit,
        processes=True,  # Use separate processes
    )

    # Create a Dask client
    client = Client(cluster)
    logging.info(
        f"Starting Dask cluster with {num_workers} workers, {memory_per_worker_gb}GB per worker"
    )

    # Display dashboard link
    dashboard_link = client.dashboard_link
    if dashboard_link:
        logging.info(f"Dask dashboard available at: {dashboard_link}")

    # First, send the gc module to all workers
    def import_gc():
        import gc

        return True

    client.run(import_gc)
    logging.info("Initialized garbage collection on all workers")

    # Process parameters one at a time to avoid overloading the scheduler
    success_count = 0

    # Group parameter directories by parameter values if requested
    if group_params:
        logging.info("Grouping parameter directories with identical parameters...")

        # Get a list of all parameter directories with their parameter values
        futures = []
        for param_dir in param_dirs:
            future = client.submit(extract_parameters, param_dir)
            futures.append((param_dir, future))

        # Collect results
        param_info = {}
        for param_dir, future in futures:
            try:
                params = future.result()
                if params:
                    param_info[param_dir] = params
            except Exception as e:
                logging.error(f"Error extracting parameters from {param_dir}: {e}")

        # Group directories by parameter hash
        grouped_dirs = {}
        for param_dir, params in param_info.items():
            param_hash = create_param_hash(params)
            if param_hash:
                if param_hash not in grouped_dirs:
                    grouped_dirs[param_hash] = {"params": params, "dirs": []}
                grouped_dirs[param_hash]["dirs"].append(param_dir)

        # Log grouping results
        total_groups = len(grouped_dirs)
        multi_dir_groups = sum(
            1 for group in grouped_dirs.values() if len(group["dirs"]) > 1
        )
        logging.info(
            f"Found {total_groups} unique parameter sets, {multi_dir_groups} with multiple directories"
        )

        for param_hash, group in grouped_dirs.items():
            if len(group["dirs"]) > 1:
                dir_names = [d.name for d in group["dirs"]]
                logging.info(
                    f"Group {param_hash[:8]}: {len(dir_names)} directories: {dir_names}"
                )

        # Calculate batch_size if not provided
        if batch_size is None:
            # Process a reasonable number of groups at once
            batch_size = min(num_workers * 2, len(grouped_dirs))

        # Process grouped directories in batches
        logging.info(f"Using batch size: {batch_size} for grouped processing")

        # Convert to list for batch processing
        groups_list = list(grouped_dirs.values())

        for i in range(0, len(groups_list), batch_size):
            batch = groups_list[i : i + batch_size]
            logging.info(
                f"Processing batch {i // batch_size + 1} with {len(batch)} parameter groups"
            )

            # Submit jobs for this batch
            futures = []
            for group_data in batch:
                # Skip if first directory already processed
                group_hash = create_param_hash(group_data["params"])
                output_param_dir = output_dir / f"grouped_{group_hash[:8]}"

                if output_param_dir.exists() and list(
                    output_param_dir.glob("*.parquet")
                ):
                    logging.info(
                        f"Parameter group {group_hash[:8]} already processed, skipping"
                    )
                    success_count += 1
                    continue

                # Submit the job
                future = client.submit(
                    process_grouped_parameter_directories, group_data, output_dir
                )
                futures.append(future)

            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    # Force garbage collection after each result
                    client.run(import_gc)
                except Exception as e:
                    logging.error(f"Error processing future: {e}")

            # Force garbage collection between batches
            client.run(import_gc)
            gc.collect()  # Also collect garbage in the main process
    else:
        # Original processing without grouping
        # Calculate batch_size if not provided
        if batch_size is None:
            # Process a reasonable number of param_dirs at once (but not all)
            batch_size = min(num_workers * 2, len(param_dirs))

        # Log the batch size being used
        logging.info(f"Using batch size: {batch_size}")

        for i in range(0, len(param_dirs), batch_size):
            batch = param_dirs[i : i + batch_size]
            logging.info(
                f"Processing batch {i // batch_size + 1} with {len(batch)} parameter directories"
            )

            # Submit jobs for this batch
            futures = []
            for param_dir in batch:
                # Skip already processed directories
                output_param_dir = output_dir / param_dir.name
                if output_param_dir.exists() and list(
                    output_param_dir.glob("*.parquet")
                ):
                    logging.info(
                        f"Parameter directory {param_dir.name} already processed, skipping"
                    )
                    success_count += 1
                    continue

                # Submit the job
                future = client.submit(
                    process_parameter_directory, param_dir, output_dir
                )
                futures.append(future)

            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    # Force garbage collection after each result
                    client.run(import_gc)
                except Exception as e:
                    logging.error(f"Error processing future: {e}")

            # Force garbage collection between batches
            client.run(import_gc)
            gc.collect()  # Also collect garbage in the main process

    # Clean up
    client.close()
    cluster.close()

    return success_count


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Import garbage collector for better memory management
    import gc

    # Ensure input and output directories exist
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logging.error(f"Input directory {input_dir} does not exist")
        return 1

    # Find all parameter directories (each containing one or more run_*.zarr directories)
    param_dirs = [d for d in input_dir.glob("*") if d.is_dir()]

    if not param_dirs:
        logging.error(f"No parameter directories found in {input_dir}")
        return 1

    logging.info(f"Found {len(param_dirs)} parameter directories to process")

    # Process parameter directories
    if args.no_parallel:
        # Sequential processing
        if args.group_params:
            # Group directories by parameter values
            logging.info("Grouping parameter directories with identical parameters...")
            grouped_dirs = group_param_dirs_by_params(param_dirs)

            total_groups = len(grouped_dirs)
            multi_dir_groups = sum(
                1 for group in grouped_dirs.values() if len(group["dirs"]) > 1
            )
            logging.info(
                f"Found {total_groups} unique parameter sets, {multi_dir_groups} with multiple directories"
            )

            # Process each group
            success_count = 0
            for param_hash, group_data in grouped_dirs.items():
                # Skip if already processed
                output_param_dir = output_dir / f"grouped_{param_hash[:8]}"
                if output_param_dir.exists() and list(
                    output_param_dir.glob("*.parquet")
                ):
                    logging.info(
                        f"Parameter group {param_hash[:8]} already processed, skipping"
                    )
                    success_count += 1
                    continue

                result = process_grouped_parameter_directories(group_data, output_dir)
                if result:
                    success_count += 1

                # Force garbage collection
                gc.collect()

            # Report results
            logging.info(
                f"Processed {success_count}/{total_groups} parameter groups successfully"
            )
            return 0 if success_count == total_groups else 1
        else:
            # Original sequential processing without grouping
            success_count = 0
            for param_dir in param_dirs:
                # Skip already processed directories
                output_param_dir = output_dir / param_dir.name
                if output_param_dir.exists() and list(
                    output_param_dir.glob("*.parquet")
                ):
                    logging.info(
                        f"Parameter directory {param_dir.name} already processed, skipping"
                    )
                    success_count += 1
                    continue

                result = process_parameter_directory(param_dir, output_dir)
                if result:
                    success_count += 1

                # Force garbage collection
                gc.collect()
    else:
        # Parallel processing using Dask
        success_count = process_with_dask(
            param_dirs,
            output_dir,
            args.num_workers,
            args.memory_per_worker,
            args.batch_size,
            args.group_params,
        )

    # Report results
    logging.info(
        f"Processed {success_count}/{len(param_dirs)} parameter directories successfully"
    )

    return 0 if success_count == len(param_dirs) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
