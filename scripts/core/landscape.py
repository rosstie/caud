import numpy as np
import copy
from ..utils.utils import (
    Item,
    getRandomInteractionMatrixNKmodel,
    getSubcombinationPayoffValues,
    reformatBinState,
    spinFlipBinState,
    perturbSubcombinationPayoffs,
    getStatesPayoffsNKmodel,
)


class FitnessLandscape:
    """
    Represents a fitness landscape for agent groups in an NK model.

    Args:
        parameters (dict): A dictionary containing the parameters for the FitnessLandscape object.
            - numberOfAgentGroups (int): The number of agent groups.
            - N_NKmodel (int): The N parameter for the NK model.
            - K_NKmodel (int): The K parameter for the NK model.
            - scaleNKFitness (bool): Whether to scale the NK fitness values.

    Attributes:
        numberOfAgentGroups (int): The number of agent groups.
        N_NKmodel (int): The N parameter for the NK model.
        K_NKmodel (int): The K parameter for the NK model.
        scaleNKFitness (bool): Whether to scale the NK fitness values.
        landscape (list): The landscape generated based on the parameters.
        allItems (list): A list of all items in the landscape.
        itemWithMaximumPayoff (str): The item with the maximum payoff in the landscape.
        numberOfItems (int): The number of items in the landscape.
    """

    def __init__(self, parameters):
        """
        Initializes a FitnessLandscapes object.

        Args:
            parameters (dict): A dictionary containing the parameters for the Landscapes object.
                - numberOfAgentGroups (int): The number of agent groups.
                - N_NKmodel (int): The N parameter for the NK model.
                - K_NKmodel (int): The K parameter for the NK model.
                - scaleNKFitness (bool): Whether to scale the NK fitness values.

        Attributes:
            numberOfAgentGroups (int): The number of agent groups.
            N_NKmodel (int): The N parameter for the NK model.
            K_NKmodel (int): The K parameter for the NK model.
            scaleNKFitness (bool): Whether to scale the NK fitness values.
            landscape (list): The landscape generated based on the parameters.
            allItems (list): A list of all items in the landscape.
            itemWithMaximumPayoff (str): The item with the maximum payoff in the landscape.
            numberOfItems (int): The number of items in the landscape.
        """
        self.numberOfAgentGroups = parameters["numberOfAgentGroups"]
        self.N_NKmodel = parameters["N_NKmodel"]
        self.K_NKmodel = parameters["K_NKmodel"]
        self.scaleNKFitness = parameters["scaleNKFitness"]
        self.positionsPerGroup = 2**self.N_NKmodel
        self.totalPositions = self.positionsPerGroup * self.numberOfAgentGroups

        # storage for NK structures
        self.interactionMatrices = []
        self.subcombinationPayoffs = []

        self.landscape = (
            self.getLandscape()
        )  # sets up the landscape for the first time and is altered by landscapeChange function
        self.allItems, self.itemWithMaximumPayoff = (
            self.getAllItemsFromLandscape()
        )  # get all items at the start.
        self.numberOfItems = len(self.allItems)

    def getAllItemsFromLandscape(self):
        """
        Retrieves all items from the landscape.

        Returns:
            allItems (list): A list of all items in the landscape.
            maxValues (ndarray): An array containing the maximum values for each column in the landscape.
        """
        allItems = []
        numberOfItems = self.landscape.shape[1]

        for itemIdx in range(numberOfItems):
            newItem = Item(itemIdx, self.landscape[:, itemIdx])
            allItems.append(newItem)

        return allItems, np.max(self.landscape, axis=0)

    def getLandscape(self):
        """
        Returns the landscape of NK payoffs for each agent group at set up.

        Returns:
            numpy.ndarray: A 2D array of size (numberOfAgentGroups, 2**N_NKmodel) containing the NK payoffs for each agent group.
        """
        NKPayoffs = np.zeros((self.numberOfAgentGroups, 2**self.N_NKmodel))

        # Clear existing structures if they exist
        self.interactionMatrices = []
        self.subcombinationPayoffs = []

        for agentGroupIdx in range(self.numberOfAgentGroups):
            # Generate and store interaction matrix
            interaction_matrix = getRandomInteractionMatrixNKmodel(
                self.N_NKmodel, self.K_NKmodel
            )
            self.interactionMatrices.append(interaction_matrix)

            # Generate and store subcombination payoffs
            subcomb_payoffs = getSubcombinationPayoffValues(
                self.N_NKmodel, self.K_NKmodel
            )
            self.subcombinationPayoffs.append(subcomb_payoffs)

            # Generate and store NK payoffs
            NKPayoffIdx = getStatesPayoffsNKmodel(
                self.N_NKmodel, self.K_NKmodel, subcomb_payoffs, interaction_matrix
            )

            # Scale NK payoffs
            NKPayoffIdx /= np.max(NKPayoffIdx)
            NKPayoffIdx = NKPayoffIdx**8
            NKPayoffs[agentGroupIdx, :] = NKPayoffIdx
        return NKPayoffs


    def landscapeChange(self, turbulence_levels):
        """
        Updates NK landscape according to disaster impacts while preserving structure.

        Args:
            turbulence_levels (np.array): Array of turbulence values per group (0-1 scale)

        Returns:
            tuple: (overall_metrics, group_metrics)
                - overall_metrics: Array of [rmsd, correlation_change, magnitude]
                - group_metrics: 2D array of shape [n_groups, 3] with [rmsd, correlation_change, magnitude] for each group
        """
        # Validate input
        assert len(turbulence_levels) == self.numberOfAgentGroups, (
            f"Expected {self.numberOfAgentGroups} turbulence values, got {len(turbulence_levels)}"
        )
        assert np.all(turbulence_levels >= 0) and np.all(turbulence_levels <= 1), (
            f"Turbulence values must be between 0 and 1, got: {turbulence_levels}"
        )

        # Check if any impact
        if np.all(turbulence_levels <= 0):
            return (np.zeros(3), np.zeros((self.numberOfAgentGroups, 3)))

        # Store pre-disaster state for validation
        old_landscape = self.landscape.copy()

        # Initialize metrics arrays
        group_metrics = np.zeros((self.numberOfAgentGroups, 3))

        for agentGroupIdx in range(self.numberOfAgentGroups):
            # Skip if no turbulence for this group
            if turbulence_levels[agentGroupIdx] <= 0:
                continue

            # Get current state
            old_group_landscape = old_landscape[agentGroupIdx, :].copy()

            # Perturb subcombination payoffs using current turbulence level
            tau = turbulence_levels[agentGroupIdx]
            perturbed_subcombination_payoffs = perturbSubcombinationPayoffs(
                self.subcombinationPayoffs[agentGroupIdx], tau=tau
            )

            # Recalculate landscape
            new_payoffs = getStatesPayoffsNKmodel(
                self.N_NKmodel,
                self.K_NKmodel,
                perturbed_subcombination_payoffs,
                self.interactionMatrices[agentGroupIdx],
            )

            # Scale NK payoffs and normalize
            new_payoffs /= np.max(new_payoffs)
            new_payoffs = new_payoffs**8

            # Update landscape and subcombination payoffs
            self.landscape[agentGroupIdx, :] = new_payoffs
            self.subcombinationPayoffs[agentGroupIdx] = perturbed_subcombination_payoffs

            # Calculate metrics for this group
            group_metrics[agentGroupIdx, 0] = landscape_change_rmsd(
                old_group_landscape, new_payoffs
            )
            group_metrics[agentGroupIdx, 1] = landscape_correlation_change(
                old_group_landscape, new_payoffs
            )
            group_metrics[agentGroupIdx, 2] = landscape_change_magnitude(
                old_group_landscape, new_payoffs
            )

        # Update maximum payoff information
        self.allItems, self.itemWithMaximumPayoff = self.getAllItemsFromLandscape()

        # Calculate overall metrics by comparing flattened landscapes
        overall_metrics = np.array(
            [
                landscape_change_rmsd(
                    old_landscape.flatten(), self.landscape.flatten()
                ),
                landscape_correlation_change(
                    old_landscape.flatten(), self.landscape.flatten()
                ),
                landscape_change_magnitude(
                    old_landscape.flatten(), self.landscape.flatten()
                ),
            ]
        )

        return overall_metrics, group_metrics


# Mean Absolute Difference between landscapes
def landscape_change_magnitude(old_landscape, new_landscape):
    return np.mean(np.abs(old_landscape - new_landscape))


# Normalized Root Mean Square Difference, this gives information on the magnitude of the change
def landscape_change_rmsd(old_landscape, new_landscape):
    mse = np.mean((old_landscape - new_landscape) ** 2)
    return np.sqrt(mse) / np.mean(old_landscape)


# Correlation Coefficient Degradation gives the correlation between the two landscapes, higher values indicate more correlation
def landscape_correlation_change(old_landscape, new_landscape):
    return (
        1 - np.corrcoef(old_landscape, new_landscape)[0, 1]
    ) 
