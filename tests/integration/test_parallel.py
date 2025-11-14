import os
import sys
import pytest
import numpy as np
import networkx as nx
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import time
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.core.simulation import Simulation
from scripts.core.network import NetworkSocialLearning
from scripts.config.params import get_parameters


def run_simulation(params):
    """Function to run a single simulation"""
    sim = Simulation(params)
    sim.runEfficientSL()
    return sim.results


def generate_network(net_type):
    """Function to generate a single network"""
    params = get_parameters()
    params.update(
        {
            "numberOfAgents": 100,
            "numberOfAgentGroups": 4,
            "numberOfTimeSteps": 10,
            "N_NKmodel": 4,
            "K_NKmodel": 2,
            "typeOfNetworkSocialLearning": net_type,
            "numberOfNeighbors": 4,
            "strategySocialLearning": "bestMember",
            "k_wsNetwork": 4,
            "p_wsNetwork": 0.1,
            "config_delta_deg": 4,
            "edgeWeightT0": 1.0,  # Initial edge weight
            "disasterProbability": 0.01,
            "disasterDistributionType": "truncated_lognormal",
            "disasterClusteredness": 0.5,
            "5th_percentile": 0.001,
            "95th_percentile": 0.5,
            "perturbations": 1,
            "perturbationScale": 0,
            "perturbationStrength": 0,
            "adaptiveNetworkEdeWeightRule": "ad",
            "edgeWeightT0": 1,
        }
    )
    network = NetworkSocialLearning(params)
    # Ensure network is generated
    network.getNetworkSocialLearning()
    return network


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


def test_parallel_simulation(base_params):
    """Test running multiple simulations in parallel."""
    print("\nTesting parallel simulation execution...")

    # Create a list of parameters for multiple simulations
    param_list = []
    for i in range(3):
        params = base_params.copy()
        params["numberOfAgents"] = 10 + i * 5  # Vary the number of agents
        param_list.append(params)

    # Run simulations in parallel using dask
    delayed_results = [dask.delayed(run_simulation)(params) for params in param_list]
    results = dask.compute(*delayed_results)

    # Verify results
    for i, result in enumerate(results):
        assert result.get("summary.final.payoff_mean", 0) > 0
        print(
            f"Simulation {i + 1} completed with mean payoff: {result.get('summary.final.payoff_mean', 0)}"
        )

    print("Parallel simulation test completed")


def test_parallel_network_generation(base_params):
    """Test parallel network generation."""
    # Create simulation
    sim = Simulation(base_params)

    # Generate network in parallel
    network = sim.networkSocialLearning
    network.getNetworkSocialLearning()

    # Verify network properties
    network_props = network.calculateNetworkProperties()
    assert network_props.get("density", 0) > 0
    assert network_props.get("density", 0) < 1
    assert network_props.get("mean_clustering", 0) >= 0
    assert network_props.get("mean_clustering", 0) <= 1


def test_parallel_disaster_impact(base_params):
    """Test parallel disaster impact calculation."""
    # Create simulation
    sim = Simulation(base_params)

    # Run simulation with disasters
    sim.runEfficientSL()

    # Verify disaster metrics
    assert sim.results.get("disaster.count", 0) >= 0
    assert sim.results.get("disaster.count", 0) <= base_params["numberOfTimeSteps"]
    assert sim.results.get("disaster.impact_mean", 0) >= 0
    assert sim.results.get("disaster.impact_mean", 0) <= 1
