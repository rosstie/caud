import os
import sys
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.run_post_processing import load_run_data, process_parameter_directory

# Set up logging to see what's happening
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@pytest.fixture
def create_test_zarr_structure():
    """Create a temp directory with zarr file structure for testing."""
    # Create a temporary directory
    base_dir = Path(tempfile.mkdtemp())

    # Create a parameter directory structure
    param_dir = base_dir / "test_param_hash"
    param_dir.mkdir()

    # Create run directories
    run_dirs = []
    for i in range(3):  # Create 3 runs
        run_dir = param_dir / f"run_{i}.zarr"
        run_dir.mkdir()
        run_dirs.append(run_dir)

        # Create time series data
        time_series_dir = run_dir / "time_series"
        time_series_dir.mkdir()

        # Create a time series dataset
        time_series_data = {
            "payoffs": (["time", "group"], np.random.rand(5, 2)),
            "learning_rate": (["time"], np.random.rand(5)),
        }
        coords = {"time": np.arange(5), "group": np.arange(2)}
        time_series = xr.Dataset(data_vars=time_series_data, coords=coords)
        time_series.to_zarr(time_series_dir)

        # Create summary data
        summary_dir = run_dir / "summary"
        summary_dir.mkdir()

        # Create a summary dataset
        summary_data = {
            "payoff_mean": ([], float(np.random.rand())),
            "payoff_std": ([], float(np.random.rand() * 0.2)),
            "item_diversity": ([], float(1 + np.random.rand())),
            "group_payoff_diff": ([], float(np.random.rand() * 0.2)),
        }
        summary = xr.Dataset(data_vars=summary_data)
        summary.to_zarr(summary_dir)

        # Create disaster data
        disaster_dir = run_dir / "disaster"
        disaster_dir.mkdir()

        # Create a disaster dataset
        disaster_data = {
            "count": ([], int(np.random.randint(1, 5))),
            "impact_mean": ([], float(np.random.rand() * 0.5)),
            "impact_std": ([], float(np.random.rand() * 0.2)),
            "recovery_rate": ([], float(0.5 + np.random.rand() * 0.5)),
        }
        disaster = xr.Dataset(data_vars=disaster_data)
        disaster.to_zarr(disaster_dir)

        # Create network data
        network_dir = run_dir / "network"
        network_dir.mkdir()

        # Create a network dataset
        network_data = {
            "density": ([], float(np.random.rand() * 0.5)),
            "mean_clustering": ([], float(np.random.rand() * 0.3)),
            "diameter": ([], float(2 + np.random.rand() * 2)),
            "centralization": ([], float(np.random.rand() * 0.4)),
        }
        network = xr.Dataset(data_vars=network_data)
        network.to_zarr(network_dir)

        # Create meta data
        meta_dir = run_dir / "meta"
        meta_dir.mkdir()

        # Create a meta dataset
        params_dict = {
            "numberOfAgents": 10,
            "numberOfAgentGroups": 2,
            "N_NKmodel": 4,
            "K_NKmodel": 2,
            "numberOfTimeSteps": 5,
        }

        meta_data = {
            "number_of_agents": ([], 10),
            "number_of_groups": ([], 2),
            "parameters": ([], np.array(str(params_dict), dtype=object)),
        }
        meta = xr.Dataset(data_vars=meta_data)
        meta.to_zarr(meta_dir)

    # Return base_dir and param_dir for cleanup and testing
    yield (base_dir, param_dir)

    # Cleanup
    shutil.rmtree(base_dir)


def test_load_run_data(create_test_zarr_structure):
    """Test loading data from a single run directory."""
    _, param_dir = create_test_zarr_structure
    run_dir = next(param_dir.glob("run_*.zarr"))

    # Load data from the run directory
    run_data = load_run_data(run_dir)

    # Verify the structure
    assert run_data is not None
    assert "time_series" in run_data
    assert "summary" in run_data
    assert "disaster" in run_data
    assert "network" in run_data
    assert "meta" in run_data

    # Verify data types
    assert isinstance(run_data["time_series"], xr.Dataset)
    assert isinstance(run_data["summary"], xr.Dataset)
    assert isinstance(run_data["disaster"], xr.Dataset)
    assert isinstance(run_data["network"], xr.Dataset)
    assert isinstance(run_data["meta"], xr.Dataset)

    # Verify some basic contents
    assert "payoffs" in run_data["time_series"].data_vars
    assert "learning_rate" in run_data["time_series"].data_vars
    assert "payoff_mean" in run_data["summary"].data_vars
    assert "count" in run_data["disaster"].data_vars
    assert "density" in run_data["network"].data_vars
    assert "parameters" in run_data["meta"].data_vars


def test_process_parameter_directory(create_test_zarr_structure):
    """Test processing all runs in a parameter directory."""
    base_dir, param_dir = create_test_zarr_structure

    # Create output directory
    output_dir = base_dir / "output"
    output_dir.mkdir()

    # Process the parameter directory
    result = process_parameter_directory(param_dir, output_dir)

    # Verify the result
    assert result is True

    # Verify output directory structure
    output_param_dir = output_dir / param_dir.name
    assert output_param_dir.exists()

    # Verify output files
    expected_files = [
        "time_series.parquet",
        "summary.parquet",
        "disaster.parquet",
        "network.parquet",
    ]
    for file_name in expected_files:
        file_path = output_param_dir / file_name
        assert file_path.exists(), f"Expected output file not found: {file_name}"

        # Load and check the file
        df = pd.read_parquet(file_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        # Check for parameter columns
        param_columns = [col for col in df.columns if col.startswith("param_")]
        assert len(param_columns) > 0, f"No parameter columns found in {file_name}"

        # Check specific parameter columns
        expected_params = [
            "param_numberOfAgents",
            "param_numberOfAgentGroups",
            "param_dir",
            "param_num_runs",
        ]
        for param in expected_params:
            assert param in df.columns, (
                f"Expected parameter column {param} not found in {file_name}"
            )

        # Verify num_runs is correct
        assert df["param_num_runs"].iloc[0] == 3

        # Verify directory hash is preserved
        assert df["param_dir"].iloc[0] == param_dir.name


def test_process_empty_parameter_directory(create_test_zarr_structure):
    """Test processing a parameter directory with no runs."""
    base_dir, _ = create_test_zarr_structure

    # Create empty parameter directory
    empty_param_dir = base_dir / "empty_param_dir"
    empty_param_dir.mkdir()

    # Create output directory
    output_dir = base_dir / "output"
    output_dir.mkdir()

    # Process the empty parameter directory
    result = process_parameter_directory(empty_param_dir, output_dir)

    # Verify the result is False for empty directory
    assert result is False


def test_parameter_extraction(create_test_zarr_structure):
    """Test parameter extraction from meta datasets."""
    base_dir, param_dir = create_test_zarr_structure

    # Create output directory
    output_dir = base_dir / "output"
    output_dir.mkdir()

    # Process the parameter directory
    result = process_parameter_directory(param_dir, output_dir)
    assert result is True

    # Verify parameter extraction in output files
    output_param_dir = output_dir / param_dir.name
    df = pd.read_parquet(output_param_dir / "summary.parquet")

    # Verify expected parameters were extracted
    expected_params = [
        "param_numberOfAgents",
        "param_numberOfAgentGroups",
        "param_N_NKmodel",
        "param_K_NKmodel",
    ]
    for param in expected_params:
        assert param in df.columns, f"Expected parameter {param} not extracted"

    # Verify parameter values
    assert df["param_numberOfAgents"].iloc[0] == 10
    assert df["param_numberOfAgentGroups"].iloc[0] == 2
    assert df["param_N_NKmodel"].iloc[0] == 4
    assert df["param_K_NKmodel"].iloc[0] == 2
