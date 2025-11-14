import os
import sys
import pytest
import numpy as np
import networkx as nx
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.core.simulation import Simulation
from scripts.core.network import NetworkSocialLearning
from scripts.core.landscape import FitnessLandscape
from scripts.core.disaster import DisasterModel
from scripts.config.params import get_parameters


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
    """Fixture to provide a simulation instance."""
    return Simulation(base_params)


def test_initialization(simulation, base_params):
    """Test basic initialization of simulation components."""
    # Test core components
    assert isinstance(simulation.fitnessLandscape, FitnessLandscape)
    assert isinstance(simulation.networkSocialLearning, NetworkSocialLearning)
    assert isinstance(simulation.disasters, DisasterModel)

    # Test array shapes
    assert simulation.payoffClassesAgents.shape == (base_params["numberOfAgents"],)
    assert simulation.payoffsAgentsOverTime.shape == (
        base_params["numberOfAgents"],
        base_params["numberOfTimeSteps"] + 1,
    )
    assert simulation.itemsAgentsOverTime.shape == (
        base_params["numberOfAgents"],
        base_params["numberOfTimeSteps"] + 1,
    )


def test_network_generation(base_params):
    """Test network generation for different network types."""
    # Test Erdős-Rényi network
    params_er = base_params.copy()
    params_er["typeOfNetworkSocialLearning"] = "er"
    params_er["p_erNetwork"] = 0.3  # Higher probability for connectivity

    network_er = NetworkSocialLearning(params_er)
    network_er.getNetworkSocialLearning()  # Generate network

    # Verify ER network properties
    assert nx.is_connected(network_er.networkSocialLearningNetworkX)
    network_props_er = network_er.calculateNetworkProperties()
    assert 0 < network_props_er.get("density", 0) < 1

    # Test Watts-Strogatz network
    params_ws = base_params.copy()
    params_ws["typeOfNetworkSocialLearning"] = "ws"
    params_ws["k_wsNetwork"] = 4
    params_ws["p_wsNetwork"] = 0.1

    network_ws = NetworkSocialLearning(params_ws)
    network_ws.getNetworkSocialLearning()

    # Verify WS network properties
    assert nx.is_connected(network_ws.networkSocialLearningNetworkX)
    network_props_ws = network_ws.calculateNetworkProperties()
    assert network_props_ws.get("mean_clustering", 0) > 0

    # Test Barabási-Albert network
    params_ba = base_params.copy()
    params_ba["typeOfNetworkSocialLearning"] = "ba"
    params_ba["m_baNetwork"] = 2

    network_ba = NetworkSocialLearning(params_ba)
    network_ba.getNetworkSocialLearning()

    # Verify BA network properties
    assert nx.is_connected(network_ba.networkSocialLearningNetworkX)
    network_props_ba = network_ba.calculateNetworkProperties()
    assert network_props_ba.get("assortativity", -1) > -1


def test_disaster_impact(base_params):
    """Test disaster impact on simulation."""
    params_disaster = base_params.copy()
    params_disaster["disasterProbability"] = 0.1
    params_disaster["disasterDistributionType"] = "truncated_lognormal"

    sim_disaster = Simulation(params_disaster)
    sim_disaster.runEfficientSL()

    # Verify disaster metrics
    assert (
        0
        <= sim_disaster.results.get("disaster.count", 0)
        <= params_disaster["numberOfTimeSteps"]
    )


def test_agent_behavior(simulation):
    """Test agent learning and innovation behavior."""
    simulation.runEfficientSL()

    # Test learning events
    assert hasattr(simulation, "learnedSolutions")
    assert hasattr(simulation, "innovatedSolutions")

    # Test that agents are learning and innovating
    total_learning = np.sum(simulation.learnedSolutions)
    total_innovation = np.sum(simulation.innovatedSolutions)

    # Test payoff improvement
    final_payoffs = simulation.payoffsAgentsOverTime[:, -1]
    initial_payoffs = simulation.payoffsAgentsOverTime[:, 0]
    mean_final_payoff = np.mean(final_payoffs)
    mean_initial_payoff = np.mean(initial_payoffs)

    # We expect at least some learning to occur
    assert total_learning > 0
    # We expect payoffs to improve
    assert mean_final_payoff > mean_initial_payoff


def test_edge_cases(base_params):
    """Test simulation with edge cases."""
    # Test minimal simulation
    params_min = base_params.copy()
    params_min["numberOfAgents"] = 10
    params_min["numberOfAgentGroups"] = 2

    sim_min = Simulation(params_min)
    sim_min.runEfficientSL()

    # Verify minimal simulation results
    assert sim_min.results.get("summary.final.payoff_mean", 0) > 0

    # Test single group
    params_single = base_params.copy()
    params_single["numberOfAgentGroups"] = 1

    sim_single = Simulation(params_single)
    sim_single.runEfficientSL()

    # Verify single group results
    assert sim_single.results.get("summary.final.item_diversity", 1) == 1


def test_performance(base_params):
    """Test simulation performance with larger parameters."""
    params_large = base_params.copy()
    params_large["numberOfAgents"] = 100
    params_large["numberOfTimeSteps"] = 20

    sim_large = Simulation(params_large)
    sim_large.runEfficientSL()

    # Verify that the simulation completes and produces results
    assert sim_large.results.get("summary.final.payoff_mean", 0) > 0
    assert sim_large.results.get("summary.final.item_diversity", 1) > 0
