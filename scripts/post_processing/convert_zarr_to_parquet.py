#!/usr/bin/env python3
"""
Convert raw zarr simulation data to flat parquet files.

This script directly examines zarr file structure and converts all non-timeseries data
into a flat parquet format where each row represents a single run.

It can group parameter directories with identical parameters before conversion
and allows selective extraction of specific variables to reduce memory usage.
"""

import os
import pandas as pd
import numpy as np
import zarr
import xarray as xr
from pathlib import Path
import argparse
import logging
import json
import hashlib
import ast
from typing import Dict, List, Union, Set, Optional, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("zarr2parquet")


def extract_parameters(param_dir: Path) -> Optional[Dict]:
    """
    Extract parameter values from a single run directory within the parameter directory.

    Args:
        param_dir: Path to the parameter directory

    Returns:
        Dictionary of parameter values or None if extraction fails
    """
    try:
        # Find the first run directory
        run_dirs = list(param_dir.glob("run_*.zarr"))
        if not run_dirs:
            logger.error(f"No run directories found in {param_dir}")
            return None

        run_dir = run_dirs[0]

        # Load metadata from xarray
        try:
            meta = xr.open_zarr(run_dir / "meta")

            # Extract parameters
            if "parameters" in meta:
                param_data = meta["parameters"].values.item()

                # Handle different parameter formats
                if isinstance(param_data, dict):
                    params = param_data
                elif isinstance(param_data, str):
                    # Try to evaluate string as dict
                    try:
                        params = ast.literal_eval(param_data)
                    except:
                        logger.warning(
                            f"Could not parse parameters string in {param_dir}"
                        )
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
            # If xarray approach fails, try using zarr directly
            logger.warning(f"XArray approach failed, trying zarr directly: {e}")
            meta_dir = run_dir / "meta"
            if not meta_dir.exists():
                logger.error(f"No meta directory found in {run_dir}")
                return None

            # Try to load parameters array
            try:
                params_array = zarr.open(str(meta_dir / "parameters"), mode="r")
                param_data = params_array[()]

                if isinstance(param_data, dict):
                    params = param_data
                elif isinstance(param_data, str):
                    try:
                        params = ast.literal_eval(param_data)
                    except:
                        logger.warning(f"Could not parse parameters string")
                        return None
                elif hasattr(param_data, "item"):
                    dict_data = param_data.item()
                    if isinstance(dict_data, dict):
                        params = dict_data
                    else:
                        return None
                else:
                    return None

                # Try to add number of agents/groups if available
                try:
                    num_agents_array = zarr.open(
                        str(meta_dir / "number_of_agents"), mode="r"
                    )
                    num_groups_array = zarr.open(
                        str(meta_dir / "number_of_groups"), mode="r"
                    )

                    if "numberOfAgents" not in params:
                        params["numberOfAgents"] = int(num_agents_array[()])
                    if "numberOfAgentGroups" not in params:
                        params["numberOfAgentGroups"] = int(num_groups_array[()])
                except:
                    pass

                return params
            except Exception as e2:
                logger.error(f"Failed to extract parameters using zarr approach: {e2}")
                return None
    except Exception as e:
        logger.error(f"Error extracting parameters from {param_dir}: {e}")
        return None


def create_param_hash(params: Dict) -> Optional[str]:
    """
    Create a hash from parameter values to identify identical parameter sets.

    Args:
        params: Dictionary of parameters

    Returns:
        MD5 hash of the parameter dictionary
    """
    if not params:
        return None

    # Create sorted representation of parameters for consistent hashing
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def group_param_dirs_by_params(param_dirs: List[Path]) -> Dict[str, Dict]:
    """
    Group parameter directories by their parameter values.

    Args:
        param_dirs: List of parameter directory paths

    Returns:
        Dictionary mapping parameter hashes to groups of directories
    """
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
            logger.info(f"Grouped directories with identical parameters: {dir_names}")

    return grouped_dirs


def load_xarray_data(
    run_dir: Path, selected_vars: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load data from a run directory using xarray directly.

    Args:
        run_dir: Path to the run directory
        selected_vars: Optional list of variables to extract (None for all)

    Returns:
        Dictionary of data values
    """
    data = {}
    datasets = ["meta", "summary", "disaster", "network", "recovery"]

    try:
        # Extract run_id from directory name
        run_id = run_dir.name.split(".")[0]
        data["run_id"] = run_id

        # Load each dataset
        for dataset_name in datasets:
            dataset_path = run_dir / dataset_name
            if not dataset_path.exists():
                continue

            ds = xr.open_zarr(dataset_path)

            # Filter variables if selected_vars is provided
            if selected_vars:
                vars_to_extract = [
                    v
                    for v in ds.data_vars
                    if v in selected_vars or f"{dataset_name}.{v}" in selected_vars
                ]
            else:
                vars_to_extract = list(ds.data_vars)

            # Extract each variable
            for var_name in vars_to_extract:
                try:
                    # Get raw data
                    var_data = ds[var_name].values

                    # Handle different data shapes
                    if len(var_data.shape) == 0:  # scalar
                        value = var_data.item()
                    elif var_data.shape == (1,):  # single value array
                        value = var_data[0]
                        if isinstance(value, (np.integer, np.floating, np.bool_)):
                            value = value.item()
                    else:
                        # For arrays, store as is
                        value = var_data

                    # Store with dataset prefix
                    data[f"{dataset_name}.{var_name}"] = value
                except Exception as e:
                    logger.error(
                        f"Error extracting {var_name} from {dataset_name}: {e}"
                    )

        # Optionally load time series data
        time_series_path = run_dir / "time_series"
        if time_series_path.exists() and (
            selected_vars is None
            or any(v.startswith("time_series") for v in selected_vars)
        ):
            ts = xr.open_zarr(time_series_path)

            # Filter variables if selected_vars is provided
            if selected_vars:
                ts_vars = [
                    v
                    for v in ts.data_vars
                    if v in selected_vars or f"time_series.{v}" in selected_vars
                ]
            else:
                ts_vars = list(ts.data_vars)

            # Extract each variable
            for var_name in ts_vars:
                try:
                    # Get the time series data
                    var_data = ts[var_name].values

                    # Store with dataset prefix
                    data[f"time_series.{var_name}"] = var_data
                except Exception as e:
                    logger.error(f"Error extracting {var_name} from time_series: {e}")

        return data
    except Exception as e:
        logger.error(f"Error loading data from {run_dir}: {e}")
        return {}


def process_grouped_parameter_directories(
    group_data: Dict, output_dir: Path, selected_vars: Optional[List[str]] = None
) -> Optional[Path]:
    """
    Process a group of parameter directories with identical parameters.

    Args:
        group_data: Dictionary with parameter information and directories
        output_dir: Path to output directory
        selected_vars: Optional list of variables to extract

    Returns:
        Path to output file or None if processing fails
    """
    param_dirs = group_data["dirs"]
    params = group_data["params"]

    # Use hash for the output filename
    group_hash = create_param_hash(params)
    output_file = output_dir / f"grouped_{group_hash[:8]}.parquet"

    # Skip if already processed
    if output_file.exists():
        logger.info(f"Parameter group {group_hash[:8]} already processed, skipping")
        return output_file

    logger.info(
        f"Processing parameter group: {group_hash[:8]} with {len(param_dirs)} directories"
    )
    logger.info(f"Directories in group: {[d.name for d in param_dirs]}")

    # Collect all runs
    all_run_data = []
    total_run_count = 0

    for param_dir in param_dirs:
        run_dirs = list(param_dir.glob("run_*.zarr"))
        total_run_count += len(run_dirs)

        for run_dir in run_dirs:
            run_data = load_xarray_data(run_dir, selected_vars)
            if run_data:
                # Add parameter directory and group information
                run_data["param_dir"] = param_dir.name
                run_data["param_group"] = group_hash[:8]

                # Add parameters as columns with param_ prefix
                for param_name, param_value in params.items():
                    run_data[f"param_{param_name}"] = param_value

                all_run_data.append(run_data)
                logger.info(f"Successfully loaded {param_dir.name}/{run_dir.name}")

    if not all_run_data:
        logger.error("No runs were loaded successfully from any directory in group")
        return None

    logger.info(
        f"Loaded {len(all_run_data)}/{total_run_count} runs from {len(param_dirs)} directories"
    )

    # Create DataFrame from all runs
    df = pd.DataFrame(all_run_data)

    # Create mapping file to document which directories were grouped
    mapping_file = output_dir / f"grouped_{group_hash[:8]}_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(
            {
                "group_hash": group_hash,
                "directories": [d.name for d in param_dirs],
                "parameter_count": len(params),
                "run_count": len(all_run_data),
                "parameters": params,
            },
            f,
            indent=2,
        )

    # Process the DataFrame to ensure proper data types
    for col in df.columns:
        # Get sample non-null value to determine type
        sample = df[col].dropna().iloc[0] if not df[col].isna().all() else None

        if sample is None:
            continue

        if isinstance(sample, np.ndarray):
            # Convert numpy arrays to json strings for storage
            df[col] = df[col].apply(
                lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else x
            )
        elif isinstance(sample, (np.integer, np.int64, np.int32)):
            # Convert numpy integers to Python integers
            df[col] = df[col].apply(
                lambda x: int(x)
                if isinstance(x, (np.integer, np.int64, np.int32))
                else x
            )
        elif isinstance(sample, (np.floating, np.float64, np.float32)):
            # Convert numpy floats to Python floats
            df[col] = df[col].apply(
                lambda x: float(x)
                if isinstance(x, (np.floating, np.float64, np.float32))
                else x
            )
        elif isinstance(sample, np.bool_):
            # Convert numpy booleans to Python booleans
            df[col] = df[col].apply(lambda x: bool(x) if isinstance(x, np.bool_) else x)

    # Write to parquet
    df.to_parquet(output_file)
    logger.info(f"Wrote {len(df)} rows to {output_file}")

    return output_file


def process_individual_parameter_directory(
    param_dir: Path, output_dir: Path, selected_vars: Optional[List[str]] = None
) -> Optional[Path]:
    """
    Process a single parameter directory without grouping.

    Args:
        param_dir: Path to parameter directory
        output_dir: Path to output directory
        selected_vars: Optional list of variables to extract

    Returns:
        Path to output file or None if processing fails
    """
    output_file = output_dir / f"{param_dir.name}.parquet"

    # Skip if already processed
    if output_file.exists():
        logger.info(f"Parameter directory {param_dir.name} already processed, skipping")
        return output_file

    logger.info(f"Processing parameter directory: {param_dir.name}")

    # Get all run directories
    run_dirs = list(param_dir.glob("run_*.zarr"))
    if not run_dirs:
        logger.error(f"No run directories found in {param_dir}")
        return None

    logger.info(f"Found {len(run_dirs)} runs in {param_dir}")

    # Extract parameters
    params = extract_parameters(param_dir)
    if not params:
        logger.warning(
            f"Could not extract parameters from {param_dir}, using empty params"
        )
        params = {}

    # Load data from each run
    all_run_data = []
    for run_dir in run_dirs:
        run_data = load_xarray_data(run_dir, selected_vars)
        if run_data:
            # Add parameter directory
            run_data["param_dir"] = param_dir.name

            # Add parameters as columns with param_ prefix
            for param_name, param_value in params.items():
                run_data[f"param_{param_name}"] = param_value

            all_run_data.append(run_data)
            logger.info(f"Successfully loaded {run_dir.name}")

    if not all_run_data:
        logger.error(f"No runs were loaded successfully from {param_dir}")
        return None

    # Create DataFrame from all runs
    df = pd.DataFrame(all_run_data)

    # Process the DataFrame to ensure proper data types
    for col in df.columns:
        # Get sample non-null value to determine type
        sample = df[col].dropna().iloc[0] if not df[col].isna().all() else None

        if sample is None:
            continue

        if isinstance(sample, np.ndarray):
            # Convert numpy arrays to json strings for storage
            df[col] = df[col].apply(
                lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else x
            )
        elif isinstance(sample, (np.integer, np.int64, np.int32)):
            # Convert numpy integers to Python integers
            df[col] = df[col].apply(
                lambda x: int(x)
                if isinstance(x, (np.integer, np.int64, np.int32))
                else x
            )
        elif isinstance(sample, (np.floating, np.float64, np.float32)):
            # Convert numpy floats to Python floats
            df[col] = df[col].apply(
                lambda x: float(x)
                if isinstance(x, (np.floating, np.float64, np.float32))
                else x
            )
        elif isinstance(sample, np.bool_):
            # Convert numpy booleans to Python booleans
            df[col] = df[col].apply(lambda x: bool(x) if isinstance(x, np.bool_) else x)

    # Write to parquet
    df.to_parquet(output_file)
    logger.info(f"Wrote {len(df)} rows to {output_file}")

    return output_file


def convert_all_param_dirs(
    root_dir: Path,
    output_dir: Path,
    group_params: bool = True,
    selected_vars: Optional[List[str]] = None,
) -> List[Path]:
    """
    Convert all parameter directories to parquet files, optionally grouping by parameters.

    Args:
        root_dir: Path to root directory with parameter directories
        output_dir: Path to output directory
        group_params: Whether to group directories with identical parameters
        selected_vars: Optional list of variables to extract

    Returns:
        List of paths to output files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all parameter directories
    param_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not param_dirs:
        logger.warning(f"No parameter directories found in {root_dir}")
        return []

    logger.info(f"Found {len(param_dirs)} parameter directories in {root_dir}")

    output_files = []

    if group_params:
        # Group directories by parameter values
        logger.info("Grouping parameter directories with identical parameters...")
        grouped_dirs = group_param_dirs_by_params(param_dirs)

        total_groups = len(grouped_dirs)
        multi_dir_groups = sum(
            1 for group in grouped_dirs.values() if len(group["dirs"]) > 1
        )
        logger.info(
            f"Found {total_groups} unique parameter sets, {multi_dir_groups} with multiple directories"
        )

        # Process each group
        for param_hash, group_data in grouped_dirs.items():
            output_file = process_grouped_parameter_directories(
                group_data, output_dir, selected_vars
            )
            if output_file:
                output_files.append(output_file)
    else:
        # Process each parameter directory individually
        for param_dir in param_dirs:
            output_file = process_individual_parameter_directory(
                param_dir, output_dir, selected_vars
            )
            if output_file:
                output_files.append(output_file)

    return output_files


def parse_var_list(var_list_str: Optional[str]) -> Optional[List[str]]:
    """
    Parse a comma-separated string into a list of variables.

    Args:
        var_list_str: Comma-separated string of variable names

    Returns:
        List of variable names or None if input is None
    """
    if not var_list_str:
        return None

    return [v.strip() for v in var_list_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Convert zarr files to parquet format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing parameter directories with zarr files",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for parquet files"
    )
    parser.add_argument(
        "--group-params",
        action="store_true",
        default=True,
        help="Group parameter directories with identical parameters (default: True)",
    )
    parser.add_argument(
        "--no-group-params",
        action="store_false",
        dest="group_params",
        help="Process each parameter directory individually without grouping",
    )
    parser.add_argument(
        "--selected-vars",
        type=str,
        help="Comma-separated list of variables to extract (default: all variables)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Parse selected variables
    selected_vars = parse_var_list(args.selected_vars)
    if selected_vars:
        logger.info(f"Processing only selected variables: {selected_vars}")

    logger.info(f"Converting zarr files from {args.input} to parquet in {args.output}")
    logger.info(f"Parameter grouping: {'enabled' if args.group_params else 'disabled'}")

    # Convert all parameter directories
    output_files = convert_all_param_dirs(
        Path(args.input), Path(args.output), args.group_params, selected_vars
    )

    logger.info(f"Converted to {len(output_files)} parquet files")


if __name__ == "__main__":
    main()
