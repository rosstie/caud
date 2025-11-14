import numpy as np
import pandas as pd
import time
import logging
import psutil
from .landscape import FitnessLandscape
from .network import NetworkSocialLearning
from .disaster import DisasterModel
from .metrics import MetricsCollector
from ..utils.measures import (
    calculate_recovery_metrics,
    gini_coefficient,
    calculate_entropy,
    calculate_hamming_distances,
)
from ..utils.utils import (
    safe_binary_conversion,
    convert_solutions_to_binary,
    reformatBinState,
    spinFlipBinState,
    getStatesPayoffsNKmodel,
)
from ..utils.measures import calculate_cumulative_values
from ..utils.storage import ResultsStorage
from ..config.params import get_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class Simulation:
    """
    A class representing a simulation.

    This class is responsible for running a simulation by distributing agents to groups,
    setting initial items and payoffs for agents, and iterating over time steps to update
    agent's items and payoffs based on neighboring items and payoffs.
        self.landscapeChange = parameters['landscapeChangeEpochs'] # number of landscape changes in the simulation, note always a run through on the final landscape
        groupSizeCompensation (int): The number of agents in the last group, if it is smaller than the others.
        numberOfTimeSteps (int): The number of time steps in the simulation.
        fitnessLandscape (FitnessLandscape): The fitness landscape object.
        networkSocialLearning (NetworkSocialLearning): The social learning network object.
        payoffClassesAgents (ndarray): An array to store the payoff classes of agents.
        payoffsAgentsOverTime (ndarray): An array to store the payoffs of agents over time.
        itemsAgentsOverTime (ndarray): An array to store the items of agents over time.
        uniqueItemsOverTime (ndarray): An array to store the unique items over time.
        arrivalTimesMaximumSolution (ndarray): An array to store the arrival times of the maximum solution.
        agentsNotArrived (list): A list of agents that have not arrived yet.
        N_NK (int): The N_NK model parameter.
        results (dict): A dictionary to store the simulation results.
    """

    def __init__(self, parameters):
        """
        Initializes a Simulation object.

        Args:
            parameters (dict): A dictionary containing the simulation parameters.

        Attributes:
            parameters (dict): The simulation parameters.
            numberOfAgents (int): The total number of agents in the simulation.
            distributeAgentsRandomly (bool): Flag indicating whether to distribute agents randomly.
            numberOfAgentGroup p.parameters()s (ndarray): An array to store the payoff classes of agents.
            payoffsAgentsOverTime (ndarray): An array to store the payoffs of agents over time.
            itemsAgentsOverTime (ndarray): An array to store the items of agents over time.
            uniqueItemsOverTime (ndarray): An array to store the unique items over time.
            arrivalTimesMaximumSolution (ndarray): An array to store the arrival times of the maximum solution.
            agentsNotArrived (list): A list of agents that have not arrived yet.
            N_NK (int): The N_NK model parameter.
            results (dict): A dictionary to store the simulation results.
        """
        #  print("Simulation object created")

        self.parameters = parameters
        self.numberOfAgents = parameters[
            "numberOfAgents"
        ]  # provides the number of agents in the simulation to iterate over
        self.distributeAgentsRandomly = parameters["distributeAgentsRandomly"]
        self.numberOfAgentGroups = parameters["numberOfAgentGroups"]
        self.groupSizeAgents = int(self.numberOfAgents / self.numberOfAgentGroups)
        self.groupSizeCompensation = self.numberOfAgents % self.numberOfAgentGroups
        self.numberOfTimeSteps = parameters["numberOfTimeSteps"]
        self.N_NK = parameters["N_NKmodel"]

        self.fitnessLandscape = FitnessLandscape(
            self.parameters
        )  # fitness landscape for agents class stores the landscape and items
        self.networkSocialLearning = NetworkSocialLearning(
            self.parameters
        )  # network adjacency matrix captures social learning
        self.disasters = DisasterModel(self.parameters)
        self.disasterImpacts = (
            self.disasters.disaster_impacts
        )  # array of disaster impacts for each time step
        self.turbulenceLevels, self.turbulenceGini = (
            self.disasters.generate_turbulence_levels()
        )

        # Network properties for the social learning network
        self.networkProperties = self.networkSocialLearning.calculateNetworkProperties()
        self.groupNetworkMetrics = dict()  # group network metrics for the simulation

        # Arrays to store the impacts to landscapes due to turbulence for each group and overall
        # group impacts are a 3D array with the first dimension being the agent group, the second dimension being the time step
        # and the third dimension being the array of impact metrics [rmsd, correlation, magnitude]
        self.groupImpactsActual = np.zeros(
            (self.numberOfAgentGroups, self.numberOfTimeSteps + 1, 3)
        )  # overall impacts are a 2D array with the first dimension being the time step and the
        # second dimension being the array of impact metrics [rmsd, correlation, magnitude]
        self.overallImpactsActual = np.zeros((self.numberOfTimeSteps + 1, 3))

        # These are all active states of the simulation
        self.payoffClassesAgents = np.zeros(self.numberOfAgents, dtype=int)
        self.payoffsAgentsOverTime = np.zeros(
            (self.numberOfAgents, self.numberOfTimeSteps + 1)
        )
        self.itemsAgentsOverTime = np.zeros(
            (self.numberOfAgents, self.numberOfTimeSteps + 1), dtype=int
        )
        self.uniqueItemsOverTime = np.zeros(self.numberOfTimeSteps + 1, dtype=int)
        self.arrivalTimesMaximumSolution = np.ones(self.numberOfAgents, dtype=int) * -1
        self.learnedSolutions = np.zeros(self.numberOfTimeSteps + 1, dtype=int)
        self.innovatedSolutions = np.zeros(self.numberOfTimeSteps + 1, dtype=int)
        self.agentsNotArrived = list(range(self.numberOfAgents))

        # arrays for landscape change impact from turbulence indexed by time step
        self.ImpactRmsd = np.zeros(self.numberOfTimeSteps + 1)
        self.ImpactCorr = np.zeros(self.numberOfTimeSteps + 1)
        self.ImpactMagnitude = np.zeros(self.numberOfTimeSteps + 1)

        # arrars for group measures of inequality indexed by time step
        self.rmsdGini = np.zeros(self.numberOfTimeSteps + 1)
        self.corrGini = np.zeros(self.numberOfTimeSteps + 1)
        self.magnitudeGini = np.zeros(self.numberOfTimeSteps + 1)

        # results dictionary to store the results of the simulation
        self.results = dict()

        # self.perturbations = parameters['perturbations'] # number of perturbation cycles within same regime
        # self.perturbationScale = parameters['perturbationScale'] # scale of perturbation = fraction of total agents that are perturbed
        # self.perturbationStrength = parameters['perturbationStrength'] # fraction of total number of integers in agent location that are changed

        self.adaptiveNetwork = parameters["adaptiveNetwork"]
        # print("initialization of sim class complete")
        # Adaptive Network Parameters not used in the current simulation
        if self.parameters["adaptiveNetwork"] == True:
            self.networkSocialLearningNumpyEdgeWeightsOverTime = np.zeros(
                (
                    self.numberOfAgentGroups,
                    self.perturbations,
                    self.numberOfTimeSteps + 1,
                )
            )
            self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, 0] = (
                self.networkSocialLearning.networkSocialLearningNumpyEdgeWeights
            )
            self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, 1] = (
                self.networkSocialLearning.networkSocialLearningNumpyEdgeWeights
            )

            # Network Adaptation Paramaters
            self.alpha = parameters["alpha"]  # diffusion constant
            self.beta = parameters["beta"]  # rate of adaptive edge weight change
            self.gamma = parameters["gamma"]  # pickiness of nodes
            self.dt = parameters[
                "Dt"
            ]  # Delta t is the time step for the simulation, no DT is the rate of edgeweight change not the time step for the simulation

    # Define a function to measure memory usage
    def getMemoryUsage(self):
        process = psutil.Process()
        memInfo = process.memory_info()
        return memInfo.rss / 1024 / 1024  # Convert to MB

    def setInitialItemsAndPayoffsForAgents(self):
        """
        Sets the initial items and payoffs for each agent.

        This method randomly assigns an initial item to each agent and calculates the corresponding payoff based on the fitness landscape.

        Returns:
            None
        """
        for agentIdx in range(self.numberOfAgents):
            initialItem = int(
                np.random.choice(range(self.fitnessLandscape.numberOfItems))
            )
            self.itemsAgentsOverTime[agentIdx, 0] = initialItem
            self.payoffsAgentsOverTime[agentIdx, 0] = self.fitnessLandscape.landscape[
                self.payoffClassesAgents[agentIdx], initialItem
            ]

        uniqueItems = np.unique(self.itemsAgentsOverTime[:, 0])
        self.uniqueItemsOverTime[0] = uniqueItems.shape[0]
        # print('this is the initial items and payoffs for agents', self.itemsAgentsOverTime, self.payoffsAgentsOverTime)

    def distributeAgentsToGroups(self):
        """
        Distributes agents to different groups based on the specified parameters.

        If `distributeAgentsRandomly` is True, agents are randomly assigned to groups.
        If `distributeAgentsRandomly` is False, agents are assigned to groups sequentially.

        Returns:
            None
        """
        if self.distributeAgentsRandomly:
            allAgents = list(range(self.numberOfAgents))
            for groupIdx in range(self.numberOfAgentGroups):
                if groupIdx == 0:
                    groupSize = self.groupSizeAgents + self.groupSizeCompensation
                else:
                    groupSize = self.groupSizeAgents
                for i in range(groupSize):
                    chosenIdx = np.random.choice(allAgents)
                    self.payoffClassesAgents[chosenIdx] = int(groupIdx)
                    allAgents.remove(chosenIdx)
        else:
            for groupIdx in range(self.numberOfAgentGroups):
                self.payoffClassesAgents[
                    groupIdx * self.groupSizeAgents + self.groupSizeCompensation : (
                        groupIdx + 1
                    )
                    * self.groupSizeAgents
                    + self.groupSizeCompensation
                ] = groupIdx

    def getNeighboringItemEfficient(self, agentIdx, t):
        """
        Returns the neighboring item with the highest payoff for a given agent at a given time.

        Parameters:
            agentIdx (int): The index of the agent.
            t (int): The time step.

        Returns:
            tuple: A tuple containing the neighboring item with the highest payoff and its corresponding payoff.
        """

        # get the neighbors of the agent at time t
        neighborsIdcs = np.nonzero(
            self.networkSocialLearning.networkSocialLearningNumpy[agentIdx, :]
        )[0]
        # print('this is neighborsIdcs', neighborsIdcs)

        # get the items of the neighboring agents at time t
        itemsOfNeighbors = self.itemsAgentsOverTime[:, t][neighborsIdcs]
        # ('this is itemsNeigbors', itemsOfNeighbors)

        # get the focal payoffs of the neighboring items for the agent at time t
        focalPayoffsOfneighboringItems = self.fitnessLandscape.landscape[
            self.payoffClassesAgents[agentIdx], itemsOfNeighbors
        ]
        # print('this is focalPayoffsOfneighboringItems', focalPayoffsOfneighboringItems)

        # find the item with the highest payoff among the neighboring items
        # itemWithMaxFocalPayoff = itemsOfNeighbors[np.argmax(focalPayoffsOfneighboringItems)]
        # print('this is the item with the highest payoff', itemWithMaxFocalPayoff)

        # find the highest payoff among the neighboring items
        # maxFocalPayoff = np.max(focalPayoffsOfneighboringItems)
        # print('this is Payoff of that item', maxFocalPayoff)

        # calculate the index value of the neighbour with the item with the highest payoff
        # highestPayoffIndex = np.argmax(self.fitnessLandscape.landscape[self.payoffClassesAgents[neighborsIdcs], itemsOfNeighbors])
        # print('this is highestPayoffIndex', highestPayoffIndex)

        indexItemWithMaxFocalPayoff = neighborsIdcs[
            np.argmax(focalPayoffsOfneighboringItems)
        ]
        # print('IndexitemWithMaxFocalPayoff', indexItemWithMaxFocalPayoff)

        return (
            itemsOfNeighbors[np.argmax(focalPayoffsOfneighboringItems)],
            np.max(focalPayoffsOfneighboringItems),
            indexItemWithMaxFocalPayoff,
        )

    def innovateNKmodel(self, intState):
        """
        Innovates a new state in the NK model by flipping a random bit in the binary state.

        Args:
            intState (int): The integer representation of the current state.

        Returns:
            int: The integer representation of the new state after innovation.
        """
        binState = reformatBinState(
            bin(intState), self.N_NK
        )  # reformat the binary state
        position = np.random.randint(0, self.N_NK)  # randomly select a position to flip
        newBinState = spinFlipBinState(
            binState, position
        )  # flip the bit at the selected position
        newIntState = int(newBinState, 2)  # convert the new binary state to an integer
        return newIntState

    def perturbNKmodel(self, intState):
        """
        Perturbs the NK model by flipping n states.

        Parameters:
        intState (int): The current state of the model.

        Returns:
        int: The new state after applying perturbation.
        """
        binState = reformatBinState(
            bin(intState), self.N_NK
        )  # reformat the binary state
        randInt = np.random.choice(range(self.N_NK))  # randomly choose a digit to flip
        flippedBinState = spinFlipBinState(binState, randInt)  # flip the chosen digit
        flippedIntState = int(
            flippedBinState, 2
        )  # convert the flipped binary state to an integer
        return flippedIntState

    def runEfficientSL(self):
        # print("running simulation")
        # set up landscape and agents
        self.distributeAgentsToGroups()

        # print("this is the payoff classes of agents", self.payoffClassesAgents)
        self.setInitialItemsAndPayoffsForAgents()
        # print(self.payoffClassesAgents)

        self.groupNetworkMetrics = (
            NetworkSocialLearning.calculate_group_network_metrics(
                self.networkSocialLearning.networkSocialLearningNetworkX,
                self.numberOfAgents,
                self.numberOfAgentGroups,
                self.payoffClassesAgents,
            )
        )
        # logging.info(
        # "Agents have been distributed to groups and initial items and payoffs have been set"
        # )
        # print("agent items at initialization", self.itemsAgentsOverTime[:, 0])
        # print("agent payoffs at initialization", self.payoffsAgentsOverTime[:, 0])

        for t in range(
            self.numberOfTimeSteps
        ):  # iterate over the number of time steps in the simulation
            # 1. Social Learning and Innovation step first
            for agentIdx in range(
                self.numberOfAgents
            ):  # iterate over the number of agents in the simulation i.e. state of each agent in network
                # Get the neighboring item with the highest payoff
                itemWithMaxFocalPayoff, maxFocalPayoff, maxItemNeighbourId = (
                    self.getNeighboringItemEfficient(agentIdx, t)
                )

                # Update the agent's item and payoff if the maxFocalPayoff of the neighbour is greater than the agent's current payoff
                if maxFocalPayoff > self.payoffsAgentsOverTime[agentIdx, t]:
                    # record learning event for agent and agent group
                    self.learnedSolutions[t + 1] += (
                        1  # increment the number of social learning events
                    )

                    self.itemsAgentsOverTime[agentIdx, t + 1] = (
                        itemWithMaxFocalPayoff  # update the agent's item at time t+1
                    )
                    self.payoffsAgentsOverTime[agentIdx, t + 1] = (
                        maxFocalPayoff  # update the agent's payoff at time t+1
                    )

                    if self.adaptiveNetwork:  # if true then adapt network
                        # print('this is edge weight adaptation for agent', agentIdx, 'at time step', t)
                        logging.debug(
                            "Edge weight adaptation for agent %s at time step %s",
                            agentIdx,
                            t,
                        )

                        # calls the adaptNetwork method to update the edge weights of the network stored in the networkSocialLearningNumpyEdgeWeightsOverTime array
                        self.adaptNetwork(
                            agentIdx, maxItemNeighbourId, maxFocalPayoff, t
                        )  # this function takes in update rules and adjusts network weights

                        # Additional debugging for network update tracking
                        # logging.debug('Adapted edge: Edge weights at time t = %s: %s', t, self.networkSocialLearningNumpyEdgeWeightsOverTime[:,:,t])
                        # logging.debug('Adapted edge: Edge weights for t+1 at time t = %s: %s', t, self.networkSocialLearningNumpyEdgeWeightsOverTime[:,:,t+1])

                else:
                    # try innovate the agent's item
                    innovatedItem = self.innovateNKmodel(
                        self.itemsAgentsOverTime[agentIdx, t]
                    )
                    innovatedPayoff = self.fitnessLandscape.landscape[
                        self.payoffClassesAgents[agentIdx], innovatedItem
                    ]

                    if innovatedPayoff > self.payoffsAgentsOverTime[agentIdx, t]:
                        # record innovation event for agent and agent group
                        self.innovatedSolutions[t + 1] += (
                            1  # increment the number of innovation events at time = t (stored at t+1)
                        )

                        self.itemsAgentsOverTime[agentIdx, t + 1] = innovatedItem
                        self.payoffsAgentsOverTime[agentIdx, t + 1] = innovatedPayoff

                        # logging.info('Innovated item %s for agent %s at time %s with payoff %s', innovatedItem, agentIdx, t, innovatedPayoff)

                    else:
                        # keep existing item and payoff
                        self.itemsAgentsOverTime[agentIdx, t + 1] = (
                            self.itemsAgentsOverTime[agentIdx, t]
                        )  # keep the agent's item at time t
                        self.payoffsAgentsOverTime[agentIdx, t + 1] = (
                            self.payoffsAgentsOverTime[agentIdx, t]
                        )  # keep payoff at time t

            # 2. Apply disaster impacts for current timestep t
            # print('turbulence at time step t:', t, self.turbulenceLevels[:,t])
            if t > 0 and np.any(self.turbulenceLevels[:, t] > 0):
                # print(f"turbuluence at t: {t} total level {np.sum(self.turbulenceLevels[:, t])}")

                # Apply disaster impacts to landscape and get group changes
                overall_metrics, group_metrics = self.fitnessLandscape.landscapeChange(
                    self.turbulenceLevels[:, t]
                )
                # print('this is the overall_metrics', overall_metrics)
                # print('this is the group_metrics', group_metrics)

                # Store disaster impacts
                self.overallImpactsActual[t + 1] = (
                    overall_metrics  # Store overall impacts
                )
                self.groupImpactsActual[:, t + 1] = group_metrics  # Store group impacts

                # print('this is the saved group impacts', self.groupImpactsActual[:,t+1])
                # print('this is the saved overall impacts', self.overallImpactsActual[t+1])

                # Store overall impacts
                self.ImpactRmsd[t + 1] = overall_metrics[0]
                self.ImpactCorr[t + 1] = overall_metrics[1]
                self.ImpactMagnitude[t + 1] = overall_metrics[2]

                # calculate gini coefficient for rmsd, correlation and magnitude of real impacts
                self.rmsdGini[t + 1] = gini_coefficient(group_metrics[:, 0])
                self.corrGini[t + 1] = gini_coefficient(group_metrics[:, 1])
                self.magnitudeGini[t + 1] = gini_coefficient(group_metrics[:, 2])

                # Recalculate all payoffs against new landscape
                for agentIdx in range(self.numberOfAgents):
                    # Keep items from learning/innovation but get new payoffs
                    self.payoffsAgentsOverTime[agentIdx, t + 1] = (
                        self.fitnessLandscape.landscape[
                            self.payoffClassesAgents[agentIdx],
                            self.itemsAgentsOverTime[agentIdx, t + 1],
                        ]
                    )
            else:
                # No disaster impact, set group impacts to 0
                self.groupImpactsActual[:, t + 1] = np.array([0, 0, 0])
                self.overallImpactsActual[t + 1] = np.array([0, 0, 0])

            # this statement is used to keep track of the time step at which the agent finds the maximum solution
            # when analysis results this means that an arrival time of -1 means that the agent did not find the maximum solution
            # Track arrival times (only if no disaster this timestep)
            if t > 0 and not np.any(self.turbulenceLevels[:, t] > 0):
                for agentIdx in range(self.numberOfAgents):
                    if (self.arrivalTimesMaximumSolution[agentIdx] < 0) and np.isclose(
                        self.payoffsAgentsOverTime[agentIdx, t + 1], 1.0
                    ):
                        self.arrivalTimesMaximumSolution[agentIdx] = t + 1

            # update metrics and tracking for the end of the time step,
            uniqueItems = np.unique(
                self.itemsAgentsOverTime[:, t + 1]
            )  # get the unique items in the population at time t+1
            self.uniqueItemsOverTime[t + 1] = uniqueItems.shape[
                0
            ]  # store the number of unique items in the population at time t+1

            # Section to manage saving adaptive network edge weights to results dictionary if adaptive network is enabled
            # also update the edge weights of the network for the next time step to the weights at the current time step
            if self.adaptiveNetwork:
                logging.debug(
                    "Edge weights at time t = %s: %s",
                    t,
                    self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, t],
                )
                self.results[t]["networkEdgeWeights"] = (
                    self.networkSocialLearningNumpyEdgeWeightsOverTime[
                        :, :, t + 1
                    ].copy()
                )
                logging.debug(
                    "Edge weights for t+1 at time t = %s: %s",
                    t,
                    self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, t + 1],
                )

                # check if the current time step is less than the number of time steps in the simulation
                if t < self.numberOfTimeSteps - 1:
                    # update the edge weights of the network for the next time step to the weights at the current time step
                    self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, t + 1] = (
                        self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, t]
                    )

        # calculate all group measures at the end of the simulation for both payoffsAgentsOverTime and itemsAgentsOverTime
        time_series_metrics = calculate_metrics_over_time(
            self.payoffsAgentsOverTime, self.itemsAgentsOverTime, self.numberOfTimeSteps
        )

        # Update results dictionary with all required metrics
        self.results.update(
            {
                # =============================================
                # 1. TIME SERIES DATA - Keep for detailed analysis
                # =============================================
                # Performance time series
                "time_series.payoffs.mean": time_series_metrics["payoff_mean"],
                "time_series.payoffs.std": time_series_metrics["payoff_std"],
                "time_series.payoffs.median": time_series_metrics["payoff_median"],
                "time_series.payoffs.max": time_series_metrics["payoff_max"],
                "time_series.payoffs.min": time_series_metrics["payoff_min"],
                # Item time series
                "time_series.items.unique_count": time_series_metrics["unique_items"],
                "time_series.items.entropy": time_series_metrics["item_entropy"],
                "time_series.items.hamming_mean": time_series_metrics[
                    "item_hamming_mean"
                ],
                # Learning time series
                "time_series.events.social_learning": self.learnedSolutions.copy(),
                "time_series.events.innovation": self.innovatedSolutions.copy(),
                # Impact time series (used for identifying disruption points)
                "time_series.turbulence": self.turbulenceLevels.copy(),
                "time_series.impact.rmsd": self.ImpactRmsd.copy(),
                "time_series.impact.correlation": self.ImpactCorr.copy(),
                "time_series.impact.magnitude": self.ImpactMagnitude.copy(),
                # =============================================
                # 2. SUMMARY METRICS - Good for aggregation
                # =============================================
                # Final values (t=max)
                "summary.final.payoff_mean": time_series_metrics["payoff_mean"][-1],
                "summary.final.payoff_median": time_series_metrics["payoff_median"][-1],
                "summary.final.item_diversity": time_series_metrics["unique_items"][-1],
                # Cumulative totals (sum over time)
                "summary.cumulative.payoff_mean": np.sum(
                    time_series_metrics["payoff_mean"]
                ),
                "summary.cumulative.payoff_std": calculate_cumulative_std(
                    time_series_metrics["payoff_std"]
                ),
                "summary.cumulative.payoff_median": np.sum(
                    time_series_metrics["payoff_median"]
                ),
                "summary.cumulative.unique_items": np.sum(
                    time_series_metrics["unique_items"]
                ),
                "summary.cumulative.social_learning": np.sum(self.learnedSolutions),
                "summary.cumulative.innovation": np.sum(self.innovatedSolutions),
                # Per-timestep innovation and social learning
                "summary.per_timestep.innovation": np.sum(self.innovatedSolutions)
                / self.numberOfTimeSteps,
                "summary.per_timestep.social_learning": np.sum(self.learnedSolutions)
                / self.numberOfTimeSteps,
                # Innovation vs Social Learning ratio
                "summary.ratio.innovation": (
                    np.sum(self.innovatedSolutions)
                    / (
                        np.sum(self.innovatedSolutions)
                        + np.sum(self.learnedSolutions)
                        + 1e-10
                    )
                ),
                # Average over time
                "summary.average.payoff_mean": np.mean(
                    time_series_metrics["payoff_mean"]
                ),
                "summary.average.item_diversity": np.mean(
                    time_series_metrics["unique_items"]
                ),
                # Variability over time
                "summary.variability.payoff": np.std(
                    time_series_metrics["payoff_mean"]
                ),
                "summary.variability.item_diversity": np.std(
                    time_series_metrics["unique_items"]
                ),
                # =============================================
                # 3. DISASTER-SPECIFIC METRICS
                # =============================================
                # Count of disasters and total impact
                "disaster.count": np.sum(self.ImpactRmsd > 0),
                "disaster.density": np.sum(self.ImpactRmsd > 0)
                / self.numberOfTimeSteps,
                "disaster.total_impact_rmsd": np.sum(self.ImpactRmsd),
                "disaster.total_impact_magnitude": np.sum(self.ImpactMagnitude),
                "disaster.average_impact_rmsd": np.mean(
                    self.ImpactRmsd[self.ImpactRmsd > 0]
                )
                if np.any(self.ImpactRmsd > 0)
                else 0,
                "disaster.average_impact_correlation": np.mean(
                    self.ImpactCorr[self.ImpactCorr > 0]
                )
                if np.any(self.ImpactCorr > 0)
                else 0,
                "disaster.average_impact_magnitude": np.mean(
                    self.ImpactMagnitude[self.ImpactMagnitude > 0]
                )
                if np.any(self.ImpactMagnitude > 0)
                else 0,
                "disaster.max_impact_rmsd": np.max(self.ImpactRmsd)
                if len(self.ImpactRmsd) > 0
                else 0,
                "disaster.max_impact_magnitude": np.max(self.ImpactMagnitude)
                if len(self.ImpactMagnitude) > 0
                else 0,
                # Inequality metrics
                "disaster.average_gini_rmsd": np.mean(self.rmsdGini[self.rmsdGini > 0])
                if np.any(self.rmsdGini > 0)
                else 0,
                "disaster.average_gini_correlation": np.mean(
                    self.corrGini[self.corrGini > 0]
                )
                if np.any(self.corrGini > 0)
                else 0,
                "disaster.average_gini_magnitude": np.mean(
                    self.magnitudeGini[self.magnitudeGini > 0]
                )
                if np.any(self.magnitudeGini > 0)
                else 0,
                "disaster.max_gini_rmsd": np.max(self.rmsdGini)
                if len(self.rmsdGini) > 0
                else 0,
                # =============================================
                # 4. RECOVERY METRICS (calculated specially)
                # =============================================
                "recovery.metrics": calculate_recovery_metrics(
                    time_series_metrics["payoff_mean"], self.ImpactRmsd
                ),
                # =============================================
                # 5. NETWORK PROPERTIES (static)
                # =============================================
                # Overall network properties
                "network.mean_degree": self.networkProperties["mean_degree"],
                "network.std_degree": self.networkProperties["std_degree"],
                "network.mean_clustering": self.networkProperties["mean_clustering"],
                "network.global_efficiency": self.networkProperties[
                    "global_efficiency"
                ],
                "network.density": self.networkProperties["density"],
                "network.assortativity": self.networkProperties["assortativity"],
                "network.avg_path_length": self.networkProperties["avg_path_length"],
                "network.degree_percentile_25": self.networkProperties[
                    "degree_percentile_25"
                ],
                "network.degree_percentile_50": self.networkProperties[
                    "degree_percentile_50"
                ],
                "network.degree_percentile_75": self.networkProperties[
                    "degree_percentile_75"
                ],
                # Group network properties - each array is indexed by group number
                "network.group.mean_degree": self.groupNetworkMetrics["mean_degree"],
                "network.group.mean_betweenness": self.groupNetworkMetrics[
                    "mean_betweenness"
                ],
                "network.group.mean_clustering": self.groupNetworkMetrics[
                    "mean_clustering"
                ],
                "network.group.homophily": self.groupNetworkMetrics["homophily"],
                "network.group.external_internal": self.groupNetworkMetrics[
                    "external_internal"
                ],
                "network.group.path_length_other_groups": self.groupNetworkMetrics[
                    "path_length_other_groups"
                ],
                # =============================================
                # 6. METADATA
                # =============================================
                "meta.parameters": self.parameters,
                "meta.N_NK": self.N_NK,
                "meta.K_NK": self.parameters["K_NKmodel"],
                "meta.p_er_network": self.parameters["p_erNetwork"],
                "meta.number_of_agents": self.numberOfAgents,
                "meta.number_of_groups": self.numberOfAgentGroups,
                "meta.number_of_timesteps": self.numberOfTimeSteps,
            }
        )

        #   print(f"end of simulation results dictionary {self.results}")
        if self.adaptiveNetwork:
            logging.debug(
                "Edge weights at time t = %s: %s",
                t,
                self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, t],
            )
            self.results["summary"]["networkEdgeWeights"] = (
                self.networkSocialLearningNumpyEdgeWeightsOverTime[:, :, t + 1].copy()
            )

        # Final network and edge weights
        if self.adaptiveNetwork:
            self.results["finalNetworkAdjacency"] = (
                self.networkSocialLearning.networkSocialLearningNumpy
            )
            self.results["finalNetworkEdgeWeights"] = (
                self.networkSocialLearningNumpyEdgeWeightsOverTime[
                    :, :, self.numberOfTimeSteps
                ]
            )
            self.results["initialNetwork"] = self.results[0]["networkAdjacency"]

        return self.results

    def computeNumberOfUniqueItemsAgents(self):
        """
        Computes the number of unique items for each agent.

        Returns:
            numberOfUniqueItemsAgents (numpy.ndarray): An array containing the number of unique items for each agent.
        """
        numberOfUniqueItemsAgents = np.zeros(self.numberOfAgents, dtype=int)
        for agentIdx in range(self.numberOfAgents):
            itemsAgent = self.itemsAgentsOverTime[agentIdx, :]
            uniqueItems = np.unique(itemsAgent)
            numberOfUniqueItemsAgents[agentIdx] = uniqueItems.shape[0]

        return numberOfUniqueItemsAgents


def calculate_metrics_over_time(payoffs_array, items_array, number_time_steps):
    """
    Calculate metrics directly from full 2D arrays using efficient array operations

    Args:
        payoffs_array: np.array [n_agents, n_timesteps]
        items_array: np.array [n_agents, n_timesteps]

    Returns:
        dict of 1D time series arrays for each metric
    """
    n_timesteps = payoffs_array.shape[1]
    assert number_time_steps + 1 == n_timesteps
    assert payoffs_array.shape[1] == items_array.shape[1] == number_time_steps + 1

    # Payoff statistics - each returns array of length n_timesteps
    metrics = {
        "payoff_mean": np.mean(payoffs_array, axis=0),
        "payoff_std": np.std(payoffs_array, axis=0),
        "payoff_min": np.min(payoffs_array, axis=0),
        "payoff_max": np.max(payoffs_array, axis=0),
        "payoff_median": np.median(payoffs_array, axis=0),
    }

    # Item diversity metrics
    unique_counts = np.array(
        [len(np.unique(items_array[:, t])) for t in range(n_timesteps)]
    )

    # Entropy calculation vectorized over time
    entropies = np.array(
        [calculate_entropy(items_array[:, t]) for t in range(n_timesteps)]
    )

    # Hamming distances - could sample if n_agents is large
    hamming_means = np.array(
        [
            np.mean(calculate_hamming_distances(items_array[:, t]))
            for t in range(n_timesteps)
        ]
    )

    metrics.update(
        {
            "unique_items": unique_counts,
            "item_entropy": entropies,
            "item_hamming_mean": hamming_means,
        }
    )

    return metrics


# TODO: incorporate calculation for innovation and social learning metrics in recovery windows to assess how these metrics change in relation to disruptions.


# Function to calculate recovery metrics
def calculate_recovery_metrics(
    payoff_series, impact_series, window=5, threshold=0.01, max_recovery_time=50
):
    """
    Calculate metrics related to recovery after disruptions

    Args:
        payoff_series: Array of payoffs over time
        impact_series: Array of impact values (RMSD) over time
        window: Number of time steps to look ahead for recovery percentage
        threshold: Minimum impact to consider as a disruption
        max_recovery_time: Maximum number of time steps to look ahead for full recovery

    Returns:
        Dictionary of recovery metrics
    """
    # Find disruption points
    disruption_indices = np.where(impact_series > threshold)[0]

    if len(disruption_indices) == 0:
        return {
            "count": 0,
            "avg_drop_pct": 0,
            "avg_recovery_pct": 0,
            "full_recovery_rate": 0,
            "avg_time_to_recovery": 0,
            "window": window,
            "threshold": threshold,
            "max_recovery_time": max_recovery_time,
            "message": "No disruptions occurred",
        }

    drops = []
    recoveries = []
    full_recoveries = 0
    times_to_recovery = []

    for idx in disruption_indices:
        # Skip if too close to beginning
        if idx <= 0:
            continue

        # Get pre and post disruption payoffs
        pre_value = payoff_series[idx - 1]
        post_value = payoff_series[idx]

        # Skip if no actual drop occurred
        if post_value >= pre_value:
            continue

        # Calculate percentage drop
        drop_pct = (pre_value - post_value) / pre_value * 100
        drops.append(drop_pct)

        # Check recovery after window time steps
        if idx + window < len(payoff_series):
            recovery_value = payoff_series[idx + window]
            recovery_pct = min(
                ((recovery_value - post_value) / (pre_value - post_value)) * 100, 100
            )
            recoveries.append(recovery_pct)

            # Check if full recovery occurred within the window
            if recovery_value >= pre_value:
                full_recoveries += 1
                times_to_recovery.append(window)  # Recovery happened within the window
            else:
                # Look ahead to find when full recovery occurs
                time_to_recovery = None
                for t in range(
                    window + 1, min(idx + max_recovery_time, len(payoff_series))
                ):
                    if t < len(payoff_series) and payoff_series[t] >= pre_value:
                        time_to_recovery = t - idx
                        break

                if time_to_recovery is not None:
                    times_to_recovery.append(time_to_recovery)
                    full_recoveries += 1

    # Calculate summary statistics
    if len(drops) > 0:
        avg_time_to_recovery = (
            np.mean(times_to_recovery) if times_to_recovery else max_recovery_time
        )

        return {
            "count": len(drops),
            "avg_drop_pct": np.mean(drops),
            "avg_recovery_pct": np.mean(recoveries) if recoveries else 0,
            "full_recovery_rate": full_recoveries / len(drops),
            "avg_time_to_recovery": avg_time_to_recovery,
            "window": window,
            "threshold": threshold,
            "max_recovery_time": max_recovery_time,
            "message": "Disruptions occurred",
        }
    else:
        return {
            "count": 0,
            "avg_drop_pct": 0,
            "avg_recovery_pct": 0,
            "full_recovery_rate": 0,
            "avg_time_to_recovery": 0,
            "window": window,
            "threshold": threshold,
            "max_recovery_time": max_recovery_time,
            "message": "No disruptions occurred",
        }


def calculate_cumulative_std(std_values):
    """
    Correctly calculate cumulative standard deviation
    """
    cumulative_variance = (std_values**2).cumsum() / (np.arange(len(std_values)) + 1)
    return np.sqrt(cumulative_variance)
