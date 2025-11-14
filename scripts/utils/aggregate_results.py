#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core aggregation functions for simulation results.
These functions transform xarray/Zarr data to pandas/parquet format.
"""

# TODO: before runnign another set of simulations check if all required data are being logged stored properly

import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def aggregate_time_series(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate time series data across runs.

    Parameters:
    -----------
    runs : List[Dict[str, Any]]
        List of simulation results

    Returns:
    --------
    pd.DataFrame
        Aggregated time series data
    """
    # Combine all time series datasets - compute immediately to avoid dask overhead
    try:
        # For dask arrays, compute first then combine to reduce overhead
        if hasattr(runs[0]["time_series"], "compute"):
            runs_computed = [
                {"time_series": run["time_series"].compute()} for run in runs
            ]
            combined = xr.concat(
                [run["time_series"] for run in runs_computed], dim="run"
            )
        else:
            combined = xr.concat([run["time_series"] for run in runs], dim="run")

        # Force computation here to avoid distributed quantile issues
        combined = combined.compute() if hasattr(combined, "compute") else combined

        # Calculate statistics across runs - all in memory now
        stats = {
            "mean": combined.mean(dim="run"),
            "std": combined.std(dim="run"),
            "min": combined.min(dim="run"),
            "max": combined.max(dim="run"),
            "median": combined.median(dim="run"),
            "q25": combined.quantile(0.25, dim="run"),
            "q75": combined.quantile(0.75, dim="run"),
        }

        # Get time dimension size
        time_steps = len(combined.time)

        # Create DataFrame directly
        rows = []

        # Process each time step
        for t in range(time_steps):
            row_data = {"time": t}

            # For each statistic and metric
            for stat_name, stat_ds in stats.items():
                for var_name, var in stat_ds.data_vars.items():
                    # Handle multi-dimensional variables (e.g., payoffs with time & group dimensions)
                    if "group" in var.dims and "time" in var.dims:
                        # Average across groups for this time step
                        value = float(var.isel(time=t).mean(dim="group").values)
                    # Handle one-dimensional variables (e.g., learning_rate with only time dimension)
                    elif "time" in var.dims:
                        value = float(var.isel(time=t).values)
                    # Skip variables without time dimension
                    else:
                        continue

                    # Add to row data
                    row_data[f"{var_name}_{stat_name}"] = value

            rows.append(row_data)

        # Create DataFrame from rows
        result_df = pd.DataFrame(rows)
        return result_df

    except Exception as e:
        logging.error(f"Error in aggregate_time_series: {e}")
        import traceback

        logging.error(traceback.format_exc())
        raise


def aggregate_summary(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate summary statistics across runs.

    Parameters:
    -----------
    runs : List[Dict[str, Any]]
        List of simulation results

    Returns:
    --------
    pd.DataFrame
        Aggregated summary statistics
    """
    try:
        # For dask arrays, compute first then combine to reduce overhead
        if hasattr(runs[0]["summary"], "compute"):
            runs_computed = [{"summary": run["summary"].compute()} for run in runs]
            combined = xr.concat([run["summary"] for run in runs_computed], dim="run")
        else:
            combined = xr.concat([run["summary"] for run in runs], dim="run")

        # Force computation here to avoid distributed quantile issues
        combined = combined.compute() if hasattr(combined, "compute") else combined

        # Calculate statistics across runs
        stats = {
            "mean": combined.mean(dim="run"),
            "std": combined.std(dim="run"),
            "min": combined.min(dim="run"),
            "max": combined.max(dim="run"),
            "median": combined.median(dim="run"),
            "q25": combined.quantile(0.25, dim="run"),
            "q75": combined.quantile(0.75, dim="run"),
        }

        # Create a single DataFrame with all metrics and statistics
        data = {}

        for var in stats:
            for metric in stats[var].data_vars:
                values = stats[var][metric].values
                if np.isscalar(values) or values.ndim == 0:
                    data[f"{metric}_{var}"] = float(values)
                else:
                    data[f"{metric}_{var}"] = values[0]  # Take first value if array

        # Create a single row DataFrame with all metrics
        result_df = pd.DataFrame([data])
        return result_df

    except Exception as e:
        logging.error(f"Error in aggregate_summary: {e}")
        import traceback

        logging.error(traceback.format_exc())
        raise


def aggregate_disaster(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate disaster metrics across runs.

    Parameters:
    -----------
    runs : List[Dict[str, Any]]
        List of simulation results

    Returns:
    --------
    pd.DataFrame
        Aggregated disaster metrics
    """
    try:
        # For dask arrays, compute first then combine to reduce overhead
        if hasattr(runs[0]["disaster"], "compute"):
            runs_computed = [{"disaster": run["disaster"].compute()} for run in runs]
            combined = xr.concat([run["disaster"] for run in runs_computed], dim="run")
        else:
            combined = xr.concat([run["disaster"] for run in runs], dim="run")

        # Force computation here to avoid distributed quantile issues
        combined = combined.compute() if hasattr(combined, "compute") else combined

        # Calculate statistics across runs
        stats = {
            "mean": combined.mean(dim="run"),
            "std": combined.std(dim="run"),
            "min": combined.min(dim="run"),
            "max": combined.max(dim="run"),
            "median": combined.median(dim="run"),
            "q25": combined.quantile(0.25, dim="run"),
            "q75": combined.quantile(0.75, dim="run"),
        }

        # Create a single DataFrame with all metrics and statistics
        data = {}

        # Log available metrics for debugging
        logging.debug("Available disaster metrics:")
        for var in stats:
            for metric in stats[var].data_vars:
                logging.debug(f"  - {metric}")

        # Process each metric and statistic
        for var in stats:
            for metric in stats[var].data_vars:
                values = stats[var][metric].values
                if np.isscalar(values) or values.ndim == 0:
                    data[f"{metric}_{var}"] = float(values)
                else:
                    data[f"{metric}_{var}"] = values[0]  # Take first value if array

        # Verify Gini coefficients are present
        gini_metrics = [
            "average_gini_rmsd",
            "average_gini_correlation",
            "average_gini_magnitude",
            "max_gini_rmsd",
        ]

        for metric in gini_metrics:
            if not any(metric in key for key in data.keys()):
                logging.warning(f"Missing Gini coefficient metric: {metric}")
            else:
                logging.debug(f"Found Gini coefficient metric: {metric}")

        # Create a single row DataFrame with all metrics
        result_df = pd.DataFrame([data])
        return result_df

    except Exception as e:
        logging.error(f"Error in aggregate_disaster: {e}")
        import traceback

        logging.error(traceback.format_exc())
        raise


def aggregate_recovery(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate recovery metrics across runs.

    Parameters:
    -----------
    runs : List[Dict[str, Any]]
        List of simulation results

    Returns:
    --------
    pd.DataFrame
        Aggregated recovery metrics
    """
    # Combine all recovery datasets
    try:
        # For dask arrays, compute first then combine to reduce overhead
        if hasattr(runs[0]["recovery"], "compute"):
            runs_computed = [{"recovery": run["recovery"].compute()} for run in runs]
            combined = xr.concat([run["recovery"] for run in runs_computed], dim="run")
        else:
            combined = xr.concat([run["recovery"] for run in runs], dim="run")

        # Force computation here to avoid distributed quantile issues
        combined = combined.compute() if hasattr(combined, "compute") else combined

        # Calculate statistics across runs
        stats = {
            "mean": combined.mean(dim="run"),
            "std": combined.std(dim="run"),
            "min": combined.min(dim="run"),
            "max": combined.max(dim="run"),
            "median": combined.median(dim="run"),
            "q25": combined.quantile(0.25, dim="run"),
            "q75": combined.quantile(0.75, dim="run"),
        }

        # Create a single DataFrame with all metrics and statistics
        data = {}

        for var in stats:
            for metric in stats[var].data_vars:
                values = stats[var][metric].values
                if np.isscalar(values) or values.ndim == 0:
                    data[f"{metric}_{var}"] = float(values)
                else:
                    data[f"{metric}_{var}"] = values[0]  # Take first value if array

        # Create a single row DataFrame with all metrics
        result_df = pd.DataFrame([data])
        return result_df

    except Exception as e:
        logging.error(f"Error aggregating recovery metrics: {e}")
        return pd.DataFrame()


def aggregate_network(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate network metrics across runs.

    Parameters:
    -----------
    runs : List[Dict[str, Any]]
        List of simulation results

    Returns:
    --------
    pd.DataFrame
        Aggregated network metrics
    """
    try:
        # For dask arrays, compute first then combine to reduce overhead
        if hasattr(runs[0]["network"], "compute"):
            runs_computed = [{"network": run["network"].compute()} for run in runs]
            combined = xr.concat([run["network"] for run in runs_computed], dim="run")
        else:
            combined = xr.concat([run["network"] for run in runs], dim="run")

        # Force computation here to avoid distributed quantile issues
        combined = combined.compute() if hasattr(combined, "compute") else combined

        # Calculate statistics across runs
        stats = {
            "mean": combined.mean(dim="run"),
            "std": combined.std(dim="run"),
            "min": combined.min(dim="run"),
            "max": combined.max(dim="run"),
            "median": combined.median(dim="run"),
            "q25": combined.quantile(0.25, dim="run"),
            "q75": combined.quantile(0.75, dim="run"),
        }

        # Create a single DataFrame with all metrics and statistics
        data = {}

        for var in stats:
            for metric in stats[var].data_vars:
                values = stats[var][metric].values
                if np.isscalar(values) or values.ndim == 0:
                    data[f"{metric}_{var}"] = float(values)
                else:
                    data[f"{metric}_{var}"] = values[0]  # Take first value if array

        # Create a single row DataFrame with all metrics
        result_df = pd.DataFrame([data])
        return result_df

    except Exception as e:
        logging.error(f"Error in aggregate_network: {e}")
        import traceback

        logging.error(traceback.format_exc())
        raise


def aggregate_simulation_results(
    runs: List[Dict[str, Any]], params: Dict[str, Any], num_repetitions: int
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate simulation results across multiple runs.

    Parameters:
    -----------
    runs : List[Dict[str, Any]]
        List of simulation results
    params : Dict[str, Any]
        Parameter set used for simulation
    num_repetitions : int
        Number of repetitions/runs

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of aggregated DataFrames
    """
    # Aggregate different components
    time_series_df = aggregate_time_series(runs)
    summary_df = aggregate_summary(runs)
    disaster_df = aggregate_disaster(runs)
    recovery_df = aggregate_recovery(runs)
    network_df = aggregate_network(runs)

    # Debug information before adding parameters
    logging.debug(f"Parameters to add: {params}")
    logging.debug(
        f"Summary DataFrame shape before adding parameters: {summary_df.shape}"
    )
    logging.debug(f"Summary DataFrame columns: {summary_df.columns.tolist()[:5]}...")

    # Add parameter information directly to each DataFrame instead of concatenating
    for df_name in [
        "time_series_df",
        "summary_df",
        "disaster_df",
        "recovery_df",
        "network_df",
    ]:
        df = locals()[df_name]  # Get the DataFrame by name

        # Add parameter columns directly to the DataFrame
        for key, value in params.items():
            col_name = f"param_{key}"
            # Repeat value to match DataFrame length
            df[col_name] = pd.Series([value] * len(df))

        # Debug after adding parameters
        param_cols = [col for col in df.columns if col.startswith("param_")]
        logging.debug(
            f"{df_name} after adding parameters - shape: {df.shape}, param columns: {param_cols}"
        )
        if param_cols:
            logging.debug(f"Sample param values: {df[param_cols].iloc[0].to_dict()}")

    result = {
        "time_series": time_series_df,
        "summary": summary_df,
        "disaster": disaster_df,
        "recovery": recovery_df,
        "network": network_df,
    }

    # Final check
    for name, df in result.items():
        param_cols = [col for col in df.columns if col.startswith("param_")]
        logging.debug(
            f"Final check - {name} shape: {df.shape}, param columns: {param_cols}"
        )

    return result


def save_aggregated_results(
    aggregated_data: Dict[str, pd.DataFrame], output_dir: Union[str, Path]
) -> None:
    """
    Save aggregated results to Parquet files.

    Parameters:
    -----------
    aggregated_data : Dict[str, pd.DataFrame]
        Dictionary of aggregated DataFrames
    output_dir : Union[str, Path]
        Output directory for Parquet files
    """
    # Convert output_dir to Path if it's a string
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each DataFrame to a Parquet file
    for name, df in aggregated_data.items():
        df.to_parquet(output_dir / f"{name}.parquet")
        logging.info(f"Saved {name}.parquet")


def load_raw_data(data_dir: str, exclude_categories: List[str] = None) -> pd.DataFrame:
    """
    Load all data except time series.

    Parameters:
    -----------
    data_dir : str
        Directory containing simulation results
    exclude_categories : List[str], optional
        List of categories to exclude

    Returns:
    --------
    pd.DataFrame
        Loaded data as a DataFrame
    """
    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Iterate over all files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet"):
            # Load the DataFrame from the file
            df = pd.read_parquet(os.path.join(data_dir, filename))

            # Extract category from filename
            category = filename.split(".")[0]

            # Skip if the category is in the exclude list
            if exclude_categories and category in exclude_categories:
                continue

            # Add category column to the DataFrame
            df["category"] = category

            # Append the DataFrame to the all_data DataFrame
            all_data = pd.concat([all_data, df], ignore_index=True)

    return all_data
