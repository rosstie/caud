import networkx as nx
import numpy as np
import scripts.utils as utils
import matplotlib.pyplot as plt

class NetworkSocialLearning:
    """
    Represents a social learning network.

    Attributes:
        parameters (dict): A dictionary containing the parameters for NetworkSocialLearning.
        numberOfAgents (int): The number of agents.
        numberOfAgentGroups (int): The number of agent groups.
        numberOfNeighbors (int): The number of neighbors.
        typeOfNetworkSocialLearning (str): The type of social learning network.
        k_wsNetwork (int): The number of nearest neighbors to connect in a Watts-Strogatz network.
        p_wsNetwork (float): The probability of rewiring each edge in a Watts-Strogatz network.
        p_erNetwork (float): The probability of creating an edge between any two nodes in an Erdos-Renyi network.
        config_delta_deg (float): The angular difference between two agents in the configuration space.
        networkSocialLearningNetworkX (networkx.Graph): The generated social learning network.
        networkSocialLearningNumpy (numpy.ndarray): The social learning network represented as a numpy array.
    """

    def __init__(self, parameters):
        """
        Initializes an instance of NetworkSocialLearning.

        Args:
            parameters (dict): A dictionary containing the parameters for NetworkSocialLearning.
                - numberOfAgents (int): The number of agents.
                - numberOfAgentGroups (int): The number of agent groups.
                - numberOfNeighbors (int): The number of neighbors.
                - typeOfNetworkSocialLearning (str): The type of social learning network.
                - k_wsNetwork (int): The number of nearest neighbors to connect in a Watts-Strogatz network.
                - p_wsNetwork (float): The probability of rewiring each edge in a Watts-Strogatz network.
                - p_erNetwork (float): The probability of creating an edge between any two nodes in an Erdos-Renyi network.
                - config_delta_deg (float): The angular difference between two agents in the configuration space.
        """
        self.parameters = parameters
        self.numberOfAgents = parameters["numberOfAgents"]
        self.numberOfAgentGroups = parameters["numberOfAgentGroups"]
        self.numberOfNeighbors = parameters["numberOfNeighbors"]
        self.typeOfNetworkSocialLearning = parameters["typeOfNetworkSocialLearning"]
        self.k_wsNetwork = parameters["k_wsNetwork"]
        self.p_wsNetwork = parameters["p_wsNetwork"]
        self.p_erNetwork = parameters["p_erNetwork"]
        self.config_delta_deg = parameters["config_delta_deg"]

        self.networkSocialLearningNetworkX = self.getNetworkSocialLearning()

        # turn the graph into a adjacency matrix used throughout the simulation
        self.networkSocialLearningNumpy = np.array(
            nx.to_numpy_array(self.networkSocialLearningNetworkX)
        )  

        # create copy of networkSocialLearningNumpy to hold edge weights and multiplies by parameter edgeWeightT0
        self.networkSocialLearningNumpyEdgeWeights = (
            np.ones_like(self.networkSocialLearningNumpy) * parameters["edgeWeightT0"]
        )

    def getNetworkSocialLearning(self):
        """
        Generates and returns a network based on the specified type of social learning network.

        Returns:
            network (networkx.Graph): The generated social learning network.

        Raises:
            ValueError: If an invalid type of social learning network is specified.
        """

        if self.typeOfNetworkSocialLearning == "ws":
            network = nx.watts_strogatz_graph(
                self.numberOfAgents, self.k_wsNetwork, self.p_wsNetwork
            )
            return network

        if self.typeOfNetworkSocialLearning == "er":
            network = nx.erdos_renyi_graph(self.numberOfAgents, self.p_erNetwork)
            while not nx.is_connected(network):
                network = nx.erdos_renyi_graph(self.numberOfAgents, self.p_erNetwork)
            return network

        if self.typeOfNetworkSocialLearning == "config_delta":
            deg = [self.config_delta_deg] * self.numberOfAgents
            network = nx.configuration_model(deg)
            network.remove_edges_from(nx.selfloop_edges(network))
            while not nx.is_connected(network):
                network = nx.configuration_model(deg)
                network.remove_edges_from(nx.selfloop_edges(network))
            return network

        if self.typeOfNetworkSocialLearning == "ba":
            network = nx.barabasi_albert_graph(self.numberOfAgents, m=3)
            return network

        if self.typeOfNetworkSocialLearning == "line":
            network = nx.watts_strogatz_graph(self.numberOfAgents, k=2, p=0)
            network.remove_edge(self.numberOfAgents - 2, self.numberOfAgents - 1)
            return network

        if self.typeOfNetworkSocialLearning == "completeGraph":
            network = nx.complete_graph(self.numberOfAgents)
            return network
        else:
            raise ValueError("Invalid type of social learning network specified.")

    @staticmethod
    def calculate_group_network_metrics(
        networkSocialLearningNetworkX,
        numberOfAgents,
        numberOfAgentGroups,
        payoffClassesAgents,
    ):
        """
        Calculate group-level network metrics

        Returns:
            dict: Dictionary of network metrics arrays, each of shape [n_groups]
                where the entry is the mean value of the metric for that group.
        """
        G = networkSocialLearningNetworkX

        # Initialize output dictionary with empty arrays
        metrics = {
            "mean_degree": np.zeros((numberOfAgentGroups)),
            "mean_betweenness": np.zeros((numberOfAgentGroups)),
            "mean_clustering": np.zeros((numberOfAgentGroups)),
            "homophily": np.zeros((numberOfAgentGroups)),
            "external_internal": np.zeros((numberOfAgentGroups)),
            "path_length_other_groups": np.zeros((numberOfAgentGroups)),
        }

        # Calculate agent-level metrics
        degree_dict = dict(G.degree())
        betweenness_dict = nx.betweenness_centrality(G)
        clustering_dict = nx.clustering(G)

        # Initialize counters and temporary storage
        group_counts = np.zeros(numberOfAgentGroups)
        agent_metrics = {
            metric: [[] for _ in range(numberOfAgentGroups)]
            for metric in ["degree", "betweenness", "clustering"]
        }

        # Group agents by their group
        agents_by_group = [[] for _ in range(numberOfAgentGroups)]
        for agent_idx in range(numberOfAgents):
            group_idx = payoffClassesAgents[agent_idx]
            agents_by_group[group_idx].append(agent_idx)
            group_counts[group_idx] += 1

            # Store basic metrics
            agent_metrics["degree"][group_idx].append(degree_dict[agent_idx])
            agent_metrics["betweenness"][group_idx].append(betweenness_dict[agent_idx])
            agent_metrics["clustering"][group_idx].append(clustering_dict[agent_idx])

        # Calculate homophily and external-internal metrics
        for group_idx in range(numberOfAgentGroups):
            if len(agents_by_group[group_idx]) == 0:
                continue

            # For each agent in this group
            homophily_values = []
            ext_int_values = []
            path_length_values = []

            for agent_idx in agents_by_group[group_idx]:
                # Get neighbors
                neighbors = list(G.neighbors(agent_idx))
                if not neighbors:
                    continue

                # Count same-group neighbors
                same_group_neighbors = sum(
                    1 for n in neighbors if payoffClassesAgents[n] == group_idx
                )

                # Calculate homophily (% same group)
                homophily = same_group_neighbors / len(neighbors) if neighbors else 0
                homophily_values.append(homophily)

                # Calculate external-internal index
                ext_int = (
                    (len(neighbors) - same_group_neighbors) / same_group_neighbors
                    if same_group_neighbors
                    else np.inf
                )
                if np.isfinite(ext_int):
                    ext_int_values.append(ext_int)

                # Calculate path length to other groups
                if nx.is_connected(G):
                    other_group_agents = [
                        a
                        for a in range(numberOfAgents)
                        if payoffClassesAgents[a] != group_idx
                    ]
                    if other_group_agents:
                        path_lengths = [
                            nx.shortest_path_length(G, agent_idx, other)
                            for other in other_group_agents
                            if nx.has_path(G, agent_idx, other)
                        ]
                        if path_lengths:
                            path_length_values.append(np.mean(path_lengths))

            # Store the group-level metrics
            for metric_name, values in [
                ("mean_degree", agent_metrics["degree"][group_idx]),
                ("mean_betweenness", agent_metrics["betweenness"][group_idx]),
                ("mean_clustering", agent_metrics["clustering"][group_idx]),
            ]:
                if values:
                    metrics[metric_name][group_idx] = np.mean(values)

            # Store homophily and external-internal metrics
            if homophily_values:
                metrics["homophily"][group_idx] = np.mean(homophily_values)
            if ext_int_values:
                metrics["external_internal"][group_idx] = np.mean(ext_int_values)
            if path_length_values:
                metrics["path_length_other_groups"][group_idx] = np.mean(
                    path_length_values
                )

        return metrics

    def calculateNetworkProperties(self):
        """
        Calculates and returns network properties.

        Returns:
            dict: A dictionary containing the following network properties:
                - averageDegree (float): The average degree of the network.
                - averageClusteringCoefficient (float): The average clustering coefficient of the network.
                - averageShortestPathLength (float): The average shortest path length of the network.
        """
        G = self.networkSocialLearningNetworkX
        try:
            # Safely calculate degree values
            degree_values = list(dict(G.degree()).values())
            if not degree_values:  # Empty network
                return None

            # Network-level metrics with error handling
            stats = {}
            try:
                stats["mean_degree"] = np.mean(degree_values)
                stats["std_degree"] = np.std(degree_values)
                stats["mean_clustering"] = np.mean(list(nx.clustering(G).values()))
                stats["global_efficiency"] = nx.global_efficiency(G)
                stats["density"] = nx.density(G)
                stats["assortativity"] = nx.degree_assortativity_coefficient(G)
                stats["avg_path_length"] = (
                    nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan
                )

                # Add percentiles for degree distribution
                for p in [25, 50, 75]:
                    stats[f"degree_percentile_{p}"] = np.percentile(degree_values, p)

            except Exception as e:
                print(f"Error calculating some network statistics: {e}")
                # Fill missing statistics with NaN
                for key in [
                    "mean_degree",
                    "std_degree",
                    "mean_clustering",
                    "global_efficiency",
                    "density",
                    "assortativity",
                    "avg_path_length",
                    "degree_percentile_25",
                    "degree_percentile_50",
                    "degree_percentile_75",
                ]:
                    if key not in stats:
                        stats[key] = np.nan

            return stats

        except Exception as e:
            print(f"Error in calculate_network_stats: {e}")
            return None

    def rewireNetwork(self, agentGroupIdx, agentGroupPayoffs, agentGroupSolutions):
        """
        Rewires the network based on the agent group payoffs and solutions.

        Args:
            agentGroupIdx (int): The index of the agent group.
            agentGroupPayoffs (numpy.ndarray): The payoffs for the agent group.
            agentGroupSolutions (numpy.ndarray): The solutions for the agent group.
        """
        pass
