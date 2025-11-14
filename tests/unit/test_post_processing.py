import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.core.simulation import Simulation
from scripts.config.params import get_parameters
from scripts.utils.aggregate_results import aggregate_simulation_results

# Set up logging to see what's happening
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@pytest.fixture
def base_params():
    """Fixture to provide base parameters for tests."""
    params = get_parameters()
    params.update(
        {
            "numberOfAgents": 10,
            "numberOfAgentGroups": 2,
            "numberOfTimeSteps": 5,
            "N_NKmodel": 4,
            "K_NKmodel": 3,  # Maximum valid K for N=4 is N-1=3
            "numberOfNeighbors": 4,
            "distributeAgentsRandomly": False,
            "adaptiveNetwork": False,
            # Network configuration
            "typeOfNetworkSocialLearning": "er",  # Default to Erdős-Rényi
            "p_erNetwork": 0.3,
            "k_wsNetwork": 4,
            "p_wsNetwork": 0.1,
            "m_baNetwork": 2,
            "config_delta_deg": 0.1,  # Angular difference between agents
            "edgeWeightT0": 1.0,  # Initial edge weight
            # Disaster model parameters
            "disasterProbability": 0.01,
            "disasterDistributionType": "truncated_lognormal",
            "5th_percentile": 0.001,
            "95th_percentile": 0.5,
            "disasterClusteredness": 0.5,
        }
    )
    return params


@pytest.fixture
def simulation(base_params):
    """Fixture to provide a simulation instance with results."""
    sim = Simulation(base_params)
    sim.runEfficientSL()
    return sim


def test_basic_metrics(simulation):
    """Test calculation of basic metrics."""
    # Test mean payoff
    assert simulation.results.get("summary.final.payoff_mean", 0) > 0
    assert simulation.results.get("summary.final.payoff_mean", 0) <= 1

    # Test standard deviation
    assert simulation.results.get("summary.final.payoff_std", 0) >= 0

    # Test Gini coefficient
    assert simulation.results.get("summary.final.payoff_gini", 0) >= 0
    assert simulation.results.get("summary.final.payoff_gini", 0) <= 1


def test_group_metrics(simulation, base_params):
    """Test calculation of group-level metrics."""
    # Test group diversity
    assert simulation.results.get("summary.final.item_diversity", 1) >= 1
    # Note: Item diversity can be greater than the number of agent groups
    # because different agents within the same group can have different items
    assert simulation.results.get("summary.final.item_diversity", 1) > 0

    # Test group payoff differences
    assert simulation.results.get("summary.final.group_payoff_diff", 0) >= 0


def test_network_metrics(simulation):
    """Test calculation of network metrics."""
    # Test network properties
    network_props = simulation.networkSocialLearning.calculateNetworkProperties()
    assert network_props.get("density", 0) > 0
    assert network_props.get("density", 0) < 1
    assert network_props.get("mean_clustering", 0) >= 0
    assert network_props.get("mean_clustering", 0) <= 1


def test_disaster_metrics(simulation, base_params):
    """Test calculation of disaster metrics."""
    # Test disaster counts
    assert simulation.results.get("disaster.count", 0) >= 0
    assert (
        simulation.results.get("disaster.count", 0) <= base_params["numberOfTimeSteps"]
    )

    # Test disaster impact
    assert simulation.results.get("disaster.impact_mean", 0) >= 0
    assert simulation.results.get("disaster.impact_mean", 0) <= 1


def test_export_results():
    # Create a simulation with minimal parameters
    params = {
        "numberOfAgents": 10,
        "numberOfAgentGroups": 2,
        "numberOfTimeSteps": 5,
        "N_NKmodel": 4,
        "K_NKmodel": 2,
        "p_erNetwork": 0.3,
        "distributeAgentsRandomly": False,
        "adaptiveNetwork": False,
        "landscapeChangeEpochs": 1,
        "disasterType": "random",
        "disasterMagnitude": 0.1,
        "disasterFrequency": 0.2,
        "scaleNKFitness": 1.0,
        "alpha": 0.1,
        "beta": 0.1,
        "gamma": 0.1,
        "Dt": 0.1,
        "numberOfNeighbors": 4,
        "typeOfNetworkSocialLearning": "er",
        "k_wsNetwork": 4,
        "p_wsNetwork": 0.1,
        "m_baNetwork": 2,
        "config_delta_deg": 0.1,
        "edgeWeightT0": 1.0,
        "disasterProbability": 0.01,
        "disasterDistributionType": "truncated_lognormal",
        "5th_percentile": 0.001,
        "95th_percentile": 0.5,
        "disasterClusteredness": 0.5,
    }

    sim = Simulation(params)

    # Run the simulation
    results = sim.runEfficientSL()

    # Convert numpy arrays to lists for DataFrame compatibility
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    # Create DataFrame from results
    df = pd.DataFrame([results])

    # Check essential columns exist
    essential_columns = [
        "time_series.payoffs.mean",
        "time_series.payoffs.std",
        "summary.final.payoff_mean",
        "summary.final.item_diversity",
    ]
    for col in essential_columns:
        assert col in df.columns, f"Missing essential column: {col}"

    # Check data types
    assert isinstance(df["time_series.payoffs.mean"].iloc[0], (list, np.ndarray))
    assert isinstance(df["time_series.payoffs.std"].iloc[0], (list, np.ndarray))
    assert isinstance(df["summary.final.payoff_mean"].iloc[0], (int, float, np.number))
    assert isinstance(
        df["summary.final.item_diversity"].iloc[0], (int, float, np.number)
    )

    # Check values are within expected ranges
    assert df["summary.final.payoff_mean"].iloc[0] >= 0
    assert df["summary.final.payoff_mean"].iloc[0] <= 1
    assert df["summary.final.item_diversity"].iloc[0] >= 0
    assert df["summary.final.item_diversity"].iloc[0] <= sim.N_NK
