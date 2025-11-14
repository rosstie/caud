def parameters():
    """
    Returns a dictionary of parameters for the simulation.

    Returns:
        dict: A dictionary containing the following parameters:
            - numberOfAgents (int): The number of agents in the simulation.
            - distributeAgentsRandomly (bool): Whether to distribute agents randomly.
            - numberOfAgentGroups (int): The number of agent groups.
            - numberOfNeighbors (int): The number of neighbors for each agent.
            - numberOfTimeSteps (int): The number of time steps in the simulation.
            - strategySocialLearning (str): The strategy for social learning.
            - N_NKmodel (int): The N parameter for the NK model.
            - K_NKmodel (int): The K parameter for the NK model.
            - scaleNKFitness (bool): Whether to scale NK fitness.
            - typeOfNetworkSocialLearning (str): The type of network for social learning.
            - k_wsNetwork (int): The k parameter for the Watts-Strogatz network.
            - p_wsNetwork (float): The p parameter for the Watts-Strogatz network.
            - p_erNetwork (float): The p parameter for the Erdos-Renyi network.
            - config_delta_deg (int): The delta parameter for the configuration model network.
            - disasterProbability (float): The probability of a disaster used in a binomial process to determine if a disaster occurs at each time step.
            - disasterDistributionType (str): The type of distribution used to model disaster impacts.
            - disasterClusteredness (float): The clusteredness of the disasters.
            - 5th_percentile (float): The 5th percentile of the disaster impacts.
            - 95th_percentile (float): The 95th percentile of the disaster impacts.
    """

    p = {}
    p["numberOfAgents"] = [100]
    p["distributeAgentsRandomly"] = True
    p["numberOfAgentGroups"] = [1, 2, 3, 5, 8, 13, 21]
    p["numberOfNeighbors"] = [4]
    p["numberOfTimeSteps"] = [400]
    p["strategySocialLearning"] = "bestMember"
    p["N_NKmodel"] = 15
    p["K_NKmodel"] = [0, 7]
    p["scaleNKFitness"] = True  # scale the NK fitness values
    p["typeOfNetworkSocialLearning"] = (
        "er"  # ['er', 'ws', 'ba, 'config_delta', 'completeGraph', 'line']
    )
    p["k_wsNetwork"] = (
        4  # number of nearest neighbors to connect in a Watts-Strogatz network
    )
    p["p_wsNetwork"] = (
        0.1  # probability of rewiring each edge in a Watts-Strogatz network
    )
    p["p_erNetwork"] = [
        0.04,
        0.08,
        0.12,
        0.20,
        0.32,
        0.52,
        0.76,
    ]  # probability of creating an edge between any two nodes in an Erdos-Renyi network
    p["config_delta_deg"] = [
        4
    ]  # ,12,28,60] # angular difference between two agents in the configuration space
    p["disasterProbability"] = 0 # [0.01,0.04]  # low = 1/100 high = 1/25  probability of a disaster i.e. a landscape change at  each time step i.e. 0.1 means 1/10 chance of landscape change each time step
    p["disasterDistributionType"] = "beta"  # 'truncated_lognormal' or 'beta'
    p["disasterClusteredness"] = 0 #[0, 0.5 ]  # level of clustering of disasters across different agent groups, exponential decay, 0 means uniform impacts across groups, > 0  means increasing clustered impacts, capped by total impact
    p["5th_percentile"] = 0 # [0.001]  # low = 0.001 # high = 0.001 value between 0 and 1 for 5th percentile of the distribution for disaster impacts i.e. how much the landscape changes as
    p["95th_percentile"] = 0 #[0.5, 0.65]  # low = 0.5 , high = 0.65 value between 0 and 1 for 95th percentile of the distribution for disaster impacts i.e. how much the landscape changes

    #################### unused parameters ############################
    # p['landscapeChangeStrength'] = [0.01,0.8,0.4,0.2,0.1,0.05] # strength of landscape change
    p["perturbations"] = [
        1
    ]  # number of perturbation cycles within same regime, 0 means no perturbation i.e. off
    p["perturbationScale"] = [
        0
    ]  # strength of perturbation = fraction of total agents that are perturbed
    p["perturbationStrength"] = [
        0
    ]  # fraction of total number of integers in agent location that are changed

    p["adaptiveNetwork"] = False  # whether to adapt the network
    p["adaptiveNetworkEdeWeightRule"] = (
        "ad"  # ['ad', 've' ] rule that defines how edge weight is 'ad' adaptive diffusion, 've' is visco-elastic
    )
    p["edgeWeightT0"] = 1  # initial edge weight strength

    return p


def get_parameters(override_params=None):
    """
    Get default parameters for the simulation, with optional overrides.

    Args:
        override_params (dict): Optional dictionary of parameter values to override defaults

    Returns:
        dict: Complete set of parameters for the simulation
    """
    default_params = {
        "numberOfAgents": 10,
        "numberOfAgentGroups": 2,
        "numberOfTimeSteps": 20,
        "N_NKmodel": 15,
        "K_NKmodel": 7,
        "p_erNetwork": 0.04,
        "distributeAgentsRandomly": True,
        "adaptiveNetwork": False,
        "numberOfNeighbors": 4,
        "strategySocialLearning": "bestMember",
        "typeOfNetworkSocialLearning": "er",
        "k_wsNetwork": 4,
        "p_wsNetwork": 0.1,
        "config_delta_deg": 4,
        "scaleNKFitness": True,
        # Disaster model parameters (matching DisasterModel expectations)
        "disasterProbability": 0.01,
        "disasterDistributionType": "beta",
        "disasterClusteredness": 0.5,
        "5th_percentile": 0.001,
        "95th_percentile": 0.5,
        # Perturbation parameters not used in this version
        "perturbations": 1,
        "perturbationScale": 0,
        "perturbationStrength": 0,
        # Network adaptation parameters not used in this version
        "adaptiveNetworkEdeWeightRule": "ad",
        "edgeWeightT0": 1,
    }

    if override_params:
        default_params.update(override_params)

    return default_params


# Export the function
__all__ = ["get_parameters"]
