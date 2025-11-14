#!/usr/bin/env python3
"""
Convert zarr time series data to parquet format.

This script extracts time series data from zarr files and converts it to parquet format.
Time series data is handled differently than scalar data since each run has potentially
thousands of time points.
"""

import os
import pandas as pd
import numpy as np
import zarr
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("zarr_timeseries_to_parquet")


def extract_time_series_data(zarr_path):
    """
    Extract time series data from a zarr file.

    Parameters:
    -----------
    zarr_path : Path
        Path to the zarr directory

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the time series data
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        logger.error(f"Zarr path does not exist: {zarr_path}")
        return None

    # Check if time_series directory exists
    time_series_path = zarr_path / "time_series"
    if not time_series_path.exists():
        logger.error(f"Time series directory not found in {zarr_path}")
        return None

    # Extract run_id from directory name
    run_id = zarr_path.name.split(".")[0]

    # Get all time series arrays
    array_paths = [
        p
        for p in time_series_path.glob("*")
        if p.is_dir() and not p.name.startswith(".")
    ]

    # Dictionary to store time series data
    ts_data = {}

    # Extract time series arrays
    for array_path in array_paths:
        array_name = array_path.name
        try:
            # Open the zarr array
            z_array = zarr.open(str(array_path), mode="r")

            # Get the array data
            if len(z_array.shape) > 1:
                # For multi-dimensional arrays, handle appropriately
                # Here we're assuming the first dimension is time
                data = z_array[:]
                if len(z_array.shape) == 2:
                    # For 2D arrays, we can handle each column separately
                    for i in range(z_array.shape[1]):
                        ts_data[f"{array_name}_{i}"] = data[:, i]
                else:
                    # For higher dimensions, we'll just flatten each time point
                    ts_data[array_name] = np.mean(
                        data, axis=tuple(range(1, len(z_array.shape)))
                    )
            else:
                # For 1D arrays, just get the data
                ts_data[array_name] = z_array[:]

        except Exception as e:
            logger.error(f"Error extracting {array_name} from time_series: {e}")

    # Create DataFrame from time series data
    if "time" in ts_data:
        # Use time as index if available
        df = pd.DataFrame(ts_data)

        # Add run_id
        df["run_id"] = run_id

        return df
    else:
        logger.error(f"Time series data does not contain 'time' array in {zarr_path}")
        return None


def extract_metadata(zarr_path):
    """
    Extract metadata from a zarr file for inclusion with time series data.

    Parameters:
    -----------
    zarr_path : Path
        Path to the zarr directory

    Returns:
    --------
    dict
        Dictionary containing metadata
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        logger.error(f"Zarr path does not exist: {zarr_path}")
        return None

    # Check if meta directory exists
    meta_path = zarr_path / "meta"
    if not meta_path.exists():
        logger.error(f"Meta directory not found in {zarr_path}")
        return None

    metadata = {}

    # Get all metadata arrays
    array_paths = [
        p for p in meta_path.glob("*") if p.is_dir() and not p.name.startswith(".")
    ]

    # Extract metadata arrays
    for array_path in array_paths:
        array_name = array_path.name
        try:
            # Open the zarr array
            z_array = zarr.open(str(array_path), mode="r")

            # Get the scalar value
            if len(z_array.shape) == 0:
                metadata[f"meta.{array_name}"] = z_array[()]
            else:
                # If it's not a scalar, take the first value
                metadata[f"meta.{array_name}"] = z_array[0]

        except Exception as e:
            logger.error(f"Error extracting {array_name} from meta: {e}")

    return metadata


def convert_parameter_dir_timeseries(param_dir, output_dir):
    """
    Convert time series data from all runs in a parameter directory to parquet format.

    Parameters:
    -----------
    param_dir : Path
        Path to the parameter directory containing run_*.zarr files
    output_dir : Path
        Path to the output directory for parquet files

    Returns:
    --------
    Path
        Path to the output parquet file
    """
    param_dir = Path(param_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the parameter directory name for output file
    param_name = param_dir.name
    output_file = output_dir / f"{param_name}_timeseries.parquet"

    # Find all run_*.zarr directories
    run_dirs = list(param_dir.glob("run_*.zarr"))
    if not run_dirs:
        logger.warning(f"No run_*.zarr directories found in {param_dir}")
        return None

    logger.info(f"Found {len(run_dirs)} runs in {param_dir}")

    # Process each run
    all_dfs = []
    for run_dir in run_dirs:
        # Extract time series data
        ts_df = extract_time_series_data(run_dir)
        if ts_df is None:
            continue

        # Extract metadata
        metadata = extract_metadata(run_dir)
        if metadata:
            # Add metadata to each row of the time series DataFrame
            for key, value in metadata.items():
                ts_df[key] = value

        # Add parameter directory name
        ts_df["param_dir"] = param_name

        all_dfs.append(ts_df)

    if not all_dfs:
        logger.warning(f"No valid time series data extracted from {param_dir}")
        return None

    # Combine all time series DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Write to parquet
    combined_df.to_parquet(output_file)
    logger.info(f"Wrote {len(combined_df)} rows to {output_file}")

    return output_file


def convert_all_param_dirs_timeseries(root_dir, output_dir):
    """
    Convert time series data from all parameter directories to parquet files.

    Parameters:
    -----------
    root_dir : Path
        Path to the root directory containing parameter directories
    output_dir : Path
        Path to the output directory for parquet files

    Returns:
    --------
    list
        List of paths to the output parquet files
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all parameter directories (direct children of root_dir)
    param_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not param_dirs:
        logger.warning(f"No parameter directories found in {root_dir}")
        return []

    logger.info(f"Found {len(param_dirs)} parameter directories in {root_dir}")

    # Convert each parameter directory
    output_files = []
    for param_dir in param_dirs:
        output_file = convert_parameter_dir_timeseries(param_dir, output_dir)
        if output_file:
            output_files.append(output_file)

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert zarr time series data to parquet format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing parameter directories with zarr files",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for parquet files"
    )

    args = parser.parse_args()

    logger.info(
        f"Converting zarr time series files from {args.input} to parquet in {args.output}"
    )

    # Convert all parameter directories
    output_files = convert_all_param_dirs_timeseries(args.input, args.output)

    logger.info(
        f"Converted time series data from {len(output_files)} parameter directories to parquet files"
    )


if __name__ == "__main__":
    main()
