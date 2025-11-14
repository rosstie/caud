import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.core.simulation import Simulation
from scripts.core.metrics import MetricsCollector
from scripts.config.params import get_parameters


@pytest.fixture
def simulation_params():
    """Fixture to provide parameters for simulation"""
    params = get_parameters()
    params.update(
        {
            "numberOfAgents": 20,
            "numberOfAgentGroups": 4,
            "numberOfTimeSteps": 10,
            "N_NKmodel": 6,
            "K_NKmodel": 2,
            "scaleNKFitness": True,
            "numberOfNeighbors": 4,
            "distributeAgentsRandomly": False,
            "adaptiveNetwork": False,
            "typeOfNetworkSocialLearning": "er",
            "p_erNetwork": 0.3,
        }
    )
    return params


@pytest.fixture
def metrics_collector(simulation_params):
    """Fixture to provide a metrics collector"""
    return MetricsCollector(
        simulation_params["numberOfAgents"], simulation_params["numberOfTimeSteps"]
    )


def test_metrics_collector_initialization(metrics_collector, simulation_params):
    """Test that metrics collector initializes correctly"""
    # Check dimensions
    assert metrics_collector.payoffs.shape == (
        simulation_params["numberOfAgents"],
        simulation_params["numberOfTimeSteps"] + 1,
    )
    assert metrics_collector.items.shape == (
        simulation_params["numberOfAgents"],
        simulation_params["numberOfTimeSteps"] + 1,
    )
    assert metrics_collector.unique_items.shape == (
        simulation_params["numberOfTimeSteps"] + 1,
    )
    assert metrics_collector.learned_solutions.shape == (
        simulation_params["numberOfTimeSteps"] + 1,
    )
    assert metrics_collector.innovated_solutions.shape == (
        simulation_params["numberOfTimeSteps"] + 1,
    )

    # Check initial values
    assert np.all(metrics_collector.payoffs == 0)
    assert np.all(metrics_collector.items == 0)
    assert np.all(metrics_collector.unique_items == 0)
    assert np.all(metrics_collector.learned_solutions == 0)
    assert np.all(metrics_collector.innovated_solutions == 0)
    assert metrics_collector.network_metrics == {}


def test_metrics_collector_step(metrics_collector, simulation_params):
    """Test that metrics collector collects step data correctly"""
    # Generate test data
    t = 5
    payoffs = np.random.random(simulation_params["numberOfAgents"])
    items = np.random.randint(0, 10, simulation_params["numberOfAgents"])
    learned = 3
    innovated = 2

    # Collect step data
    metrics_collector.collect_step(t, payoffs, items, learned, innovated)

    # Check that data was collected correctly
    assert np.array_equal(metrics_collector.payoffs[:, t], payoffs)
    assert np.array_equal(metrics_collector.items[:, t], items)
    assert metrics_collector.learned_solutions[t] == learned
    assert metrics_collector.innovated_solutions[t] == innovated

    # Check unique items calculation
    expected_unique = len(np.unique(items))
    assert metrics_collector.unique_items[t] == expected_unique


def test_metrics_collector_network(metrics_collector):
    """Test that metrics collector collects network metrics correctly"""
    # Generate test network metrics
    network_metrics = {"density": 0.5, "clustering": 0.3, "path_length": 2.5}

    # Collect network metrics
    metrics_collector.collect_network_metrics(network_metrics)

    # Check that metrics were collected correctly
    assert metrics_collector.network_metrics == network_metrics


def test_metrics_collector_results(metrics_collector, simulation_params):
    """Test that metrics collector generates results correctly"""
    # Generate test data for multiple time steps
    for t in range(simulation_params["numberOfTimeSteps"] + 1):
        payoffs = np.random.random(simulation_params["numberOfAgents"])
        items = np.random.randint(0, 10, simulation_params["numberOfAgents"])
        learned = np.random.randint(0, 5)
        innovated = np.random.randint(0, 3)

        metrics_collector.collect_step(t, payoffs, items, learned, innovated)

    # Add network metrics
    network_metrics = {"density": 0.5, "clustering": 0.3, "path_length": 2.5}
    metrics_collector.collect_network_metrics(network_metrics)

    # Get results
    results = metrics_collector.get_results()

    # Check time series metrics
    assert "time_series" in results
    assert "payoff_mean" in results["time_series"]
    assert "payoff_std" in results["time_series"]
    assert "payoff_median" in results["time_series"]
    assert "unique_items" in results["time_series"]
    assert "learned_solutions" in results["time_series"]
    assert "innovated_solutions" in results["time_series"]

    # Check network metrics
    assert "network" in results
    assert results["network"] == network_metrics

    # Check summary metrics
    assert "summary" in results
    assert "final_payoff_mean" in results["summary"]
    assert "final_payoff_std" in results["summary"]
    assert "total_learned" in results["summary"]
    assert "total_innovated" in results["summary"]
    assert "avg_unique_items" in results["summary"]


def test_export_results_to_dataframe(metrics_collector, simulation_params):
    """Test exporting results to DataFrame"""
    # Generate test data for multiple time steps
    for t in range(simulation_params["numberOfTimeSteps"] + 1):
        payoffs = np.random.random(simulation_params["numberOfAgents"])
        items = np.random.randint(0, 10, simulation_params["numberOfAgents"])
        learned = np.random.randint(0, 5)
        innovated = np.random.randint(0, 3)

        metrics_collector.collect_step(t, payoffs, items, learned, innovated)

    # Add network metrics
    network_metrics = {"density": 0.5, "clustering": 0.3, "path_length": 2.5}
    metrics_collector.collect_network_metrics(network_metrics)

    # Get results
    results = metrics_collector.get_results()

    # Create a dictionary with the results
    results_dict = {}
    for key, value in results["summary"].items():
        results_dict[key] = [value]

    for key, value in results["network"].items():
        results_dict[key] = [value]

    # Export results to DataFrame
    df = pd.DataFrame(results_dict)

    # Verify DataFrame structure
    assert "final_payoff_mean" in df.columns
    assert "final_payoff_std" in df.columns
    assert "total_learned" in df.columns
    assert "total_innovated" in df.columns
    assert "avg_unique_items" in df.columns
    assert "density" in df.columns
    assert "clustering" in df.columns
    assert "path_length" in df.columns

    # Verify data types
    assert np.issubdtype(df["final_payoff_mean"].dtype, np.number)
    assert np.issubdtype(df["final_payoff_std"].dtype, np.number)
    assert np.issubdtype(df["total_learned"].dtype, np.number)
    assert np.issubdtype(df["total_innovated"].dtype, np.number)
    assert np.issubdtype(df["avg_unique_items"].dtype, np.number)
    assert np.issubdtype(df["density"].dtype, np.number)
    assert np.issubdtype(df["clustering"].dtype, np.number)
    assert np.issubdtype(df["path_length"].dtype, np.number)
