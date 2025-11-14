import os
import sys
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import subprocess
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging to see what's happening
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@pytest.fixture
def create_zarr_test_data():
    """Create a complete zarr test dataset structure for end-to-end testing."""
    # Create a temporary directory
    base_dir = Path(tempfile.mkdtemp())
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    # Create directories
    input_dir.mkdir()
    output_dir.mkdir()

    # Create two parameter hash directories with different parameter sets
    param_hashes = ["test_param_1", "test_param_2"]

    for param_hash in param_hashes:
        param_dir = input_dir / param_hash
        param_dir.mkdir()

        # Create 3 runs for each parameter set
        for i in range(3):
            run_dir = param_dir / f"run_{i}.zarr"
            run_dir.mkdir()

            # Create subdirectories for each data type
            for data_type in ["time_series", "summary", "disaster", "network", "meta"]:
                (run_dir / data_type).mkdir()

            # Create time series data (different for each parameter)
            time_series_data = {
                "payoffs": (
                    ["time", "group"],
                    np.random.rand(5, 2)
                    * (0.5 if param_hash == "test_param_1" else 0.8),
                ),
                "learning_rate": (
                    ["time"],
                    np.random.rand(5) * (0.2 if param_hash == "test_param_1" else 0.4),
                ),
            }
            coords = {"time": np.arange(5), "group": np.arange(2)}
            time_series = xr.Dataset(data_vars=time_series_data, coords=coords)
            time_series.to_zarr(run_dir / "time_series")

            # Create summary data
            summary_data = {
                "payoff_mean": (
                    [],
                    float(
                        np.random.rand()
                        * (0.5 if param_hash == "test_param_1" else 0.8)
                    ),
                ),
                "payoff_std": ([], float(np.random.rand() * 0.2)),
                "item_diversity": ([], float(1 + np.random.rand())),
                "group_payoff_diff": ([], float(np.random.rand() * 0.2)),
            }
            summary = xr.Dataset(data_vars=summary_data)
            summary.to_zarr(run_dir / "summary")

            # Create disaster data
            disaster_data = {
                "count": ([], int(np.random.randint(1, 5))),
                "impact_mean": ([], float(np.random.rand() * 0.5)),
                "impact_std": ([], float(np.random.rand() * 0.2)),
                "recovery_rate": ([], float(0.5 + np.random.rand() * 0.5)),
            }
            disaster = xr.Dataset(data_vars=disaster_data)
            disaster.to_zarr(run_dir / "disaster")

            # Create network data
            network_data = {
                "density": ([], float(np.random.rand() * 0.5)),
                "mean_clustering": ([], float(np.random.rand() * 0.3)),
                "diameter": ([], float(2 + np.random.rand() * 2)),
                "centralization": ([], float(np.random.rand() * 0.4)),
            }
            network = xr.Dataset(data_vars=network_data)
            network.to_zarr(run_dir / "network")

            # Create meta data with parameter set specific to each directory
            if param_hash == "test_param_1":
                params_dict = {
                    "numberOfAgents": 10,
                    "numberOfAgentGroups": 2,
                    "N_NKmodel": 4,
                    "K_NKmodel": 2,
                    "disasterProbability": 0.2,
                }
            else:
                params_dict = {
                    "numberOfAgents": 20,
                    "numberOfAgentGroups": 3,
                    "N_NKmodel": 6,
                    "K_NKmodel": 3,
                    "disasterProbability": 0.4,
                }

            meta_data = {
                "number_of_agents": ([], params_dict["numberOfAgents"]),
                "number_of_groups": ([], params_dict["numberOfAgentGroups"]),
                "parameters": ([], np.array(str(params_dict), dtype=object)),
            }
            meta = xr.Dataset(data_vars=meta_data)
            meta.to_zarr(run_dir / "meta")

    yield (base_dir, input_dir, output_dir)

    # Cleanup
    shutil.rmtree(base_dir)


def verify_zarr_data_structure(input_dir):
    """Verify that the test zarr directory structure was created correctly."""
    param_dirs = list(input_dir.glob("*"))
    assert len(param_dirs) > 0, f"No parameter directories found in {input_dir}"

    for param_dir in param_dirs:
        run_dirs = list(param_dir.glob("run_*.zarr"))
        assert len(run_dirs) > 0, f"No run directories found in {param_dir}"

        for run_dir in run_dirs:
            # Check that all required directories exist
            for data_type in ["time_series", "summary", "disaster", "network", "meta"]:
                assert (run_dir / data_type).exists(), (
                    f"Missing {data_type} directory in {run_dir}"
                )

                # Verify each directory has data
                files = list((run_dir / data_type).glob("*"))
                assert len(files) > 0, f"No files found in {run_dir}/{data_type}"


def test_zarr_data_structure(create_zarr_test_data):
    """Test that our fixture correctly creates the zarr data structure."""
    _, input_dir, _ = create_zarr_test_data
    verify_zarr_data_structure(input_dir)
    logging.info("Test zarr directory structure created successfully")


def run_post_processing_script(script_path, input_dir, output_dir, debug=False):
    """Run the post-processing script and return the command result."""
    # Ensure it's an executable script
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, 0o755)  # Make it executable

    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--no-parallel",  # Use sequential processing for testing
    ]

    if debug:
        cmd.append("--log-level")
        cmd.append("DEBUG")

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print output for debugging
    if debug or result.returncode != 0:
        logging.info(f"Command: {' '.join(cmd)}")
        logging.info(f"Return code: {result.returncode}")
        logging.info(f"STDOUT: {result.stdout}")
        logging.info(f"STDERR: {result.stderr}")

    return result


def test_end_to_end_pipeline_script(create_zarr_test_data):
    """Test the end-to-end pipeline by running the script as a subprocess."""
    base_dir, input_dir, output_dir = create_zarr_test_data

    # First, verify the test data structure is correct
    verify_zarr_data_structure(input_dir)

    # Get the script path
    script_path = (
        Path(__file__).parent.parent.parent / "scripts" / "run_post_processing.py"
    )
    assert script_path.exists(), f"Script not found: {script_path}"

    # List run directories before running
    input_param_dirs = list(input_dir.glob("*"))
    for param_dir in input_param_dirs:
        run_dirs = list(param_dir.glob("run_*.zarr"))
        logging.info(
            f"Parameter directory {param_dir.name} has {len(run_dirs)} run directories"
        )

    # Run the post-processing script with debug output
    result = run_post_processing_script(script_path, input_dir, output_dir, debug=True)

    # Check that the command succeeded
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"

    # Check if output directory exists and has files
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    output_contents = list(output_dir.glob("*"))
    logging.info(f"Output directory contents: {[p.name for p in output_contents]}")

    # Verify output directory structure
    param_dirs = list(input_dir.glob("*"))
    assert len(param_dirs) > 0, "No parameter directories found"
    logging.info(
        f"Found {len(param_dirs)} parameter directories: {[p.name for p in param_dirs]}"
    )

    # Check output for each parameter directory
    for param_dir in param_dirs:
        param_name = param_dir.name
        output_param_dir = output_dir / param_name

        # Check if parameter directory was created in output
        assert output_param_dir.exists(), (
            f"Output parameter directory {param_name} not created"
        )

        # List files in the output parameter directory
        output_files = list(output_param_dir.glob("*"))
        logging.info(f"Files in {output_param_dir}: {[f.name for f in output_files]}")

        # Check for all expected output files
        expected_files = [
            "time_series.parquet",
            "summary.parquet",
            "disaster.parquet",
            "network.parquet",
        ]
        for file_name in expected_files:
            file_path = output_param_dir / file_name
            assert file_path.exists(), (
                f"Missing output file: {file_name} in {output_param_dir}"
            )
            assert file_path.stat().st_size > 0, (
                f"Empty output file: {file_name} in {output_param_dir}"
            )


def test_aggregated_data_correctness(create_zarr_test_data):
    """Test that the aggregated data contains the correct values and structure."""
    base_dir, input_dir, output_dir = create_zarr_test_data

    # Get the script path
    script_path = (
        Path(__file__).parent.parent.parent / "scripts" / "run_post_processing.py"
    )

    # Run the post-processing script with debug output
    result = run_post_processing_script(script_path, input_dir, output_dir, debug=True)
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"

    # Verify the data correctness for both parameter sets
    param_dirs = list(input_dir.glob("*"))

    for param_dir in param_dirs:
        output_param_dir = output_dir / param_dir.name
        assert output_param_dir.exists(), (
            f"Output directory {output_param_dir} not created"
        )

        # Check if summary file exists
        summary_path = output_param_dir / "summary.parquet"
        assert summary_path.exists(), f"Summary file not found: {summary_path}"
        assert summary_path.stat().st_size > 0, f"Empty summary file: {summary_path}"

        # Check the summary parquet file
        try:
            summary_df = pd.read_parquet(summary_path)
        except Exception as e:
            logging.error(f"Error reading {summary_path}: {e}")
            raise

        # Check that we have a single aggregated row
        assert len(summary_df) == 1, (
            f"Expected 1 row in summary data, got {len(summary_df)}"
        )

        # Check parameter columns
        param_columns = [col for col in summary_df.columns if col.startswith("param_")]
        assert len(param_columns) >= 5, f"Not enough parameter columns: {param_columns}"

        # Verify parameter values differ between param sets
        if param_dir.name == "test_param_1":
            assert summary_df["param_numberOfAgents"].iloc[0] == 10
            assert summary_df["param_numberOfAgentGroups"].iloc[0] == 2
            assert summary_df["param_N_NKmodel"].iloc[0] == 4
            assert summary_df["param_K_NKmodel"].iloc[0] == 2
            assert summary_df["param_disasterProbability"].iloc[0] == 0.2
        else:
            assert summary_df["param_numberOfAgents"].iloc[0] == 20
            assert summary_df["param_numberOfAgentGroups"].iloc[0] == 3
            assert summary_df["param_N_NKmodel"].iloc[0] == 6
            assert summary_df["param_K_NKmodel"].iloc[0] == 3
            assert summary_df["param_disasterProbability"].iloc[0] == 0.4

        # Verify the number of runs was tracked correctly
        assert summary_df["param_num_runs"].iloc[0] == 3

        # Check if time_series file exists
        time_series_path = output_param_dir / "time_series.parquet"
        assert time_series_path.exists(), (
            f"Time series file not found: {time_series_path}"
        )
        assert time_series_path.stat().st_size > 0, (
            f"Empty time series file: {time_series_path}"
        )

        # Check the time series parquet file
        try:
            time_series_df = pd.read_parquet(time_series_path)
        except Exception as e:
            logging.error(f"Error reading {time_series_path}: {e}")
            raise

        # Check parameter columns are included
        for col in param_columns:
            assert col in time_series_df.columns, (
                f"Parameter {col} missing from time series data"
            )


def test_parameter_differentiation_in_outputs(create_zarr_test_data):
    """Test that the parameters are correctly stored and differentiated in the outputs."""
    base_dir, input_dir, output_dir = create_zarr_test_data

    # Get the script path
    script_path = (
        Path(__file__).parent.parent.parent / "scripts" / "run_post_processing.py"
    )

    # Run the post-processing script with debug output
    result = run_post_processing_script(script_path, input_dir, output_dir, debug=True)
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"

    # Check output directories exist
    param1_dir = output_dir / "test_param_1"
    param2_dir = output_dir / "test_param_2"
    assert param1_dir.exists(), f"Output directory {param1_dir} not created"
    assert param2_dir.exists(), f"Output directory {param2_dir} not created"

    # Check summary files exist
    param1_summary = param1_dir / "summary.parquet"
    param2_summary = param2_dir / "summary.parquet"
    assert param1_summary.exists(), f"Summary file not found: {param1_summary}"
    assert param2_summary.exists(), f"Summary file not found: {param2_summary}"

    # Load both parameter sets' summary files
    try:
        param1_df = pd.read_parquet(param1_summary)
        param2_df = pd.read_parquet(param2_summary)
    except Exception as e:
        logging.error(f"Error reading summary files: {e}")
        raise

    # Verify the parameter values are different between the two sets
    assert param1_df["param_numberOfAgents"].iloc[0] == 10
    assert param2_df["param_numberOfAgents"].iloc[0] == 20

    assert param1_df["param_numberOfAgentGroups"].iloc[0] == 2
    assert param2_df["param_numberOfAgentGroups"].iloc[0] == 3

    assert param1_df["param_N_NKmodel"].iloc[0] == 4
    assert param2_df["param_N_NKmodel"].iloc[0] == 6

    assert param1_df["param_K_NKmodel"].iloc[0] == 2
    assert param2_df["param_K_NKmodel"].iloc[0] == 3

    assert param1_df["param_disasterProbability"].iloc[0] == 0.2
    assert param2_df["param_disasterProbability"].iloc[0] == 0.4

    # Verify directory hash is stored correctly
    assert param1_df["param_dir"].iloc[0] == "test_param_1"
    assert param2_df["param_dir"].iloc[0] == "test_param_2"


def test_combined_parameter_analysis(create_zarr_test_data):
    """Test loading and analyzing data from multiple parameter sets."""
    base_dir, input_dir, output_dir = create_zarr_test_data

    # Get the script path
    script_path = (
        Path(__file__).parent.parent.parent / "scripts" / "run_post_processing.py"
    )

    # Run the post-processing script with debug output
    result = run_post_processing_script(script_path, input_dir, output_dir, debug=True)
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"

    # Check output directories exist and have summary files
    param_dirs = list(output_dir.glob("*"))
    assert len(param_dirs) == 2, (
        f"Expected 2 parameter directories, got {len(param_dirs)}"
    )

    # Load all summary files to simulate a combined analysis
    summary_dfs = []

    for param_dir in param_dirs:
        summary_path = param_dir / "summary.parquet"
        assert summary_path.exists(), f"Summary file not found: {summary_path}"

        try:
            df = pd.read_parquet(summary_path)
            summary_dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {summary_path}: {e}")
            raise

    # Check that we have data from both parameter sets
    assert len(summary_dfs) == 2, (
        f"Expected 2 summary dataframes, got {len(summary_dfs)}"
    )

    # Combine into a single analysis dataframe
    combined_df = pd.concat(summary_dfs, ignore_index=True)

    # Verify we have data from both parameter sets
    assert len(combined_df) == 2, (
        f"Expected 2 rows in combined data, got {len(combined_df)}"
    )

    # Verify we can filter by parameter
    filtered_df = combined_df[combined_df["param_numberOfAgents"] == 10]
    assert len(filtered_df) == 1, (
        f"Expected 1 row with 10 agents, got {len(filtered_df)}"
    )
    assert filtered_df["param_dir"].iloc[0] == "test_param_1"

    filtered_df = combined_df[combined_df["param_numberOfAgents"] == 20]
    assert len(filtered_df) == 1, (
        f"Expected 1 row with 20 agents, got {len(filtered_df)}"
    )
    assert filtered_df["param_dir"].iloc[0] == "test_param_2"
