import numpy as np
import itertools


class Item:
    """
    Represents an item with coordinates and payoffs.
    """

    def __init__(self, coordinates, payoffs):
        """
        Initializes an instance of the class.

        Args:
            coordinates (list): A list of coordinates.
            payoffs (list): A list of payoffs.

        Returns:
            None
        """
        self.coordinates = coordinates
        self.payoffs = payoffs

    def __lt__(self, other):
        """
        Compare the current object with another object based on the first element of their payoffs.

        Args:
            other: The other object to compare with.

        Returns:
            True if the first element of the current object's payoffs is less than the first element of the other object's payoffs, False otherwise.
        """
        return self.payoffs[0] < other.payoffs[0]

    def __eq__(self, other):
        """
        Define equality comparison for Items
        """
        if not isinstance(other, Item):
            return False
        return self.coordinates == other.coordinates and np.array_equal(
            self.payoffs, other.payoffs
        )


def spinFlipBinState(binState, positionOfSpinFlip):
    """
    Flip the bit at the specified position in the binary state.
    """
    listBinState = list(binState)

    if listBinState[positionOfSpinFlip] == "0":
        listBinState[positionOfSpinFlip] = "1"
        binState = "".join(listBinState)
        return binState
    else:
        listBinState[positionOfSpinFlip] = "0"
        binState = "".join(listBinState)
        return binState


def reformatBinState(binState, N_NK):
    """
    Reformat binary state to have correct number of bits.
    """
    binState = binState.split("b")[1]
    lenBinState = len(binState)
    assert lenBinState <= N_NK, "check: something must be wrong"

    binState = "0" * (N_NK - lenBinState) + binState
    return binState


# TODO: potentially split this function into two separate functions for clearer function
# this would be getStatesPayoffsNKmodel which sets up the payoffs for each state at the start of the simulation
# and updateStatesPayoffsNKmodel which updates the payoffs for each state after a landscape change


def getStatesPayoffsNKmodel(
    N_NK, K_NK, dictSubcombinationPayoffValues, interactionMatrixNKmodel
):
    """
    Calculate the payoffs for each state in the NK model.

    Parameters:
    - N_NK (int): The number of elements in the state.
    - K_NK (int): The number of interactions for each element.
    - dictSubcombinationPayoffValues (dict): A dictionary containing the payoff values for each subcombination.

    Returns:
    - payoffsNK (numpy.ndarray): An array containing the payoffs for each state.

    """

    # interactionMatrixNKmodel = getRandomInteractionMatrixNKmodel(N_NK, K_NK) # generate a random interaction matrix for the NK model

    payoffsNK = np.zeros(
        2**N_NK
    )  # initialize an array to hold the payoffs for each state

    for intState in range(2**N_NK):  # for each possible state in the landscape
        payoffsSingleStates = np.zeros(
            N_NK
        )  # initialize an array to hold the payoffs for each element in the state
        binState = reformatBinState(
            bin(intState), N_NK
        )  # reformat the binary state to the desired length

        for digit_idx, digit in enumerate(binState):  # for each element in the state
            combinationIdcs = np.nonzero(interactionMatrixNKmodel[digit_idx, :])[
                0
            ]  # get the indices of the elements that interact with the current element
            string = digit  # initialize a string to hold the binary representation of the subcombination
            for combinationIdx in combinationIdcs:  # for each interacting element
                string += binState[
                    combinationIdx
                ]  # add the binary representation of the interacting element to the string
            payoffsSingleStates[digit_idx] = dictSubcombinationPayoffValues[
                string
            ]  # calculate the payoff for the current element

        payoffsNK[intState] = np.mean(
            payoffsSingleStates
        )  # calculate the mean payoff for the state

    return payoffsNK


def getSubcombinationPayoffValues(N_NK, K_NK):
    """
    Generates a dictionary of subcombination payoff values.

    Args:
        N_NK (int): The total number of elements.
        K_NK (int): The number of elements in each subcombination.

    Returns:
        dict: A dictionary where the keys are binary strings representing subcombinations
              and the values are randomly generated payoff values between 0 and 1.
    """

    assert 0 <= K_NK < N_NK

    dictCombinationsPayoffs = dict()
    combinations = list(itertools.product("01", repeat=K_NK + 1))

    for combination in combinations:
        string = ""
        for digit in combination:
            string += digit
        dictCombinationsPayoffs[string] = np.random.uniform(0, 1)
    # print('this is the subcombination payoff values (NKmodel.py)', dictCombinationsPayoffs)

    return dictCombinationsPayoffs


def getRandomInteractionMatrixNKmodel(N_NK, K_NK):
    """
    Create a random interaction matrix for NK model.

    Args:
        N_NK: Number of elements in the NK model
        K_NK: Number of interactions per element

    Returns:
        idcs_mat: Interaction matrix (N_NK x N_NK)
    """
    # Ensure K_NK is valid (at most N_NK-1)
    assert K_NK <= N_NK - 1, (
        f"K_NK ({K_NK}) must be at most N_NK-1 ({N_NK - 1}) in NK model"
    )

    idcs_mat = np.zeros(
        (N_NK, N_NK), dtype=int
    )  # initialize an array to hold the interaction matrix which is N x N in size
    for i in np.arange(N_NK):  # for each node in the NK model which is N in length
        idcs = list(
            range(N_NK)
        )  # create a list of indices for the nodes which will be N in length
        idcs.remove(
            i
        )  # remove the current node from the list of indices # this ensures that the node does not i
        chosen_ones = np.random.choice(
            idcs, size=K_NK, replace=False
        )  # randomly choose K nodes from the list of indices using a uniform distribution
        for j in chosen_ones:  # for each chosen node
            idcs_mat[i, j] = (
                1  # set the interaction matrix value to 1 this means that the node interacts with the chosen node and itself i.e. the diagonal is not zero
            )
    # this returns a random interaction matrix for the NK model which is N x N in size and has K interactions per node
    # print('this is the interaction matrix (NKmodel.py)', idcs_mat)
    return idcs_mat


def perturbSubcombinationPayoffs(original_payoffs, tau):
    """
    Perturb subcombination payoffs while maintaining NK landscape properties.
    This approach:
    1. Generates a new landscape with the same NK structure
    2. Interpolates between the original and new landscape

    Args:
    - original_payoffs: Original subcombination payoff dictionary
    - tau: Turbulence parameter (0-1) representing the fraction of the new landscape
          to incorporate. When tau=0, the original landscape is preserved.
          When tau=1, a completely new landscape is generated.

    Returns:
    - Perturbed subcombination payoff dictionary
    """
    # Extract N and K from the original payoffs structure
    # The length of each key represents K+1 (the element itself plus its K interactions)
    K = len(next(iter(original_payoffs.keys()))) - 1
    N = (
        K + 1
    )  # This is a simplification - in practice N should be passed as a parameter

    # Generate a new landscape with the same NK structure
    new_payoffs = getSubcombinationPayoffValues(N, K)

    # Interpolate between landscapes
    perturbed_payoffs = {}
    for config in original_payoffs.keys():
        original_value = original_payoffs[config]
        new_value = new_payoffs[config]

        # Linear interpolation between original and new landscape
        perturbed_value = (1 - tau) * original_value + tau * new_value

        perturbed_payoffs[config] = perturbed_value

    return perturbed_payoffs


def generateFilename(parameters):
    return f"simulation_agents_{parameters['numberOfAgents']}_timesteps_{parameters['numberOfTimeSteps']}_perturbations_{parameters['perturbations']}_landscapeChange_{parameters['landscapeChangeEpochs']}.h5"


def safe_binary_conversion(value):
    """
    Safely convert a value to binary string, handling edge cases.
    """
    try:
        return bin(int(value))
    except (ValueError, TypeError):
        return None


def convert_solutions_to_binary(solutions, n_bits):
    """
    Convert solution integers to binary strings of fixed length.
    """
    return [format(int(sol), f"0{n_bits}b") for sol in solutions]


__all__ = [
    "safe_binary_conversion",
    "convert_solutions_to_binary",
    "reformatBinState",
    "spinFlipBinState",
    "getRandomInteractionMatrixNKmodel",
    "getSubcombinationPayoffValues",
]
