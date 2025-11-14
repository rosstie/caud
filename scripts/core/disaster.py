import numpy as np
import logging
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from ..utils.measures import gini_coefficient


class DisasterModel:
    def __init__(self, parameters):
        """
        Initialize the DisasterModel and generates all disaster impacts at once
        Outputting a single turbulence array indexed by time step and agent group
        for the entire simulation.
        Where the value of the ijth element of the array is the turbulence level
        between 0 and 1 for the jth agent group at the ith time step.

        sum of all elements in the array at each time step should be
        equal to the disaster impact at that time step


        :param parameters: Dictionary containing model parameters
        :param num_steps: Total number of time steps in the simulation
        """
        self.parameters = parameters
        self.time_steps = self.parameters["numberOfTimeSteps"]
        self.p_event = self.parameters["disasterProbability"]
        self.distribution_type = self.parameters["disasterDistributionType"]
        self.lower_percentile = self.parameters["5th_percentile"]
        self.upper_percentile = self.parameters["95th_percentile"]
        self.N_NKmodel = self.parameters["N_NKmodel"]
        self.num_groups = self.parameters["numberOfAgentGroups"]
        self.clusteredness = self.parameters[
            "disasterClusteredness"
        ]  # default medium clustering

        self.setup_distribution()
        # print("setup distribution complete")
        self.disaster_impacts = self.generate_event_impacts()
        # print("setup disaster impacts complete")
        self.disaster_skews = self.generate_disaster_skews()
        # print("setup disaster model complete", self.disaster_skews)
        self.turbulenceLevels, self.turbulenceGini = self.generate_turbulence_levels()
        # print("setup turbulence levels complete")

    def setup_distribution(self):
        if self.distribution_type == "truncated_lognormal":

            def objective(sigma):
                mu = np.log(self.lower_percentile * self.upper_percentile) / 2
                return (
                    stats.lognorm.ppf(0.95, s=sigma, scale=np.exp(mu))
                    - self.upper_percentile
                ) ** 2

            result = minimize_scalar(objective, bounds=(0.01, 10), method="bounded")
            sigma = result.x
            mu = np.log(self.lower_percentile * self.upper_percentile) / 2

            self.distribution = stats.truncnorm(
                a=(np.log(1e-10) - mu) / sigma,
                b=(np.log(1) - mu) / sigma,
                loc=mu,
                scale=sigma,
            )
        elif self.distribution_type == "beta":

            def objective(params):
                alpha, beta = params
                return (
                    stats.beta.ppf(0.05, alpha, beta) - self.lower_percentile
                ) ** 2 + (
                    stats.beta.ppf(0.95, alpha, beta) - self.upper_percentile
                ) ** 2

            result = minimize(objective, [1, 1], method="Nelder-Mead")
            alpha, beta = result.x
            self.distribution = stats.beta(alpha, beta)
        else:
            raise ValueError(
                "Distribution type must be 'truncated_lognormal' or 'beta'"
            )

    def generate_event_impacts(self):
        """
        Generate an array of event impacts for the entire simulation at once.

        :param num_steps: Number of time steps in the simulation
        :return: Array of event impacts (0 for no event, otherwise impact value between 0 and 1)
        """
        events = (
            np.random.random(self.time_steps + 1) < self.p_event
        )  # calculates the number of events by using a bernoulli distribution with p_event as the probability of success (event) and time_steps as the number of trials
        impacts = np.zeros(self.time_steps + 1)

        if self.distribution_type == "truncated_lognormal":
            impacts[events] = np.exp(self.distribution.rvs(sum(events)))
        else:  # beta
            impacts[events] = self.distribution.rvs(sum(events))

        return impacts

    def generate_disaster_skews(self):
        """Generate skew arrays with guaranteed valid position counts"""
        skews = np.zeros((self.num_groups, len(self.disaster_impacts)))
        disaster_timesteps = np.where(
            self.disaster_impacts > 0
        )[
            0
        ]  # returns the indices of the elements that are greater than 0 in the disaster_impacts array
        # print(f"disaster timesteps: {disaster_timesteps}")
        # print(f"skews shape: {skews.shape}")

        for t in disaster_timesteps:
            # Generate weights using exponential decay
            weights = np.exp(-self.clusteredness * np.arange(self.num_groups))
            # Simple power law distribution could be used instead but less realistic
            # weights = np.power(0.5, self.clusteredness * np.arange(self.num_groups))
            # Randomly permute which groups get hit harder
            group_order = np.random.permutation(self.num_groups)
            # Normalize to sum to 1
            skews[group_order, t] = weights / np.sum(weights)
            # print(f"skews at time step {t} sum to 1 : {np.sum(skews[:, t]) == 1}")
            assert np.isclose(np.sum(skews[:, t]), 1, atol=1e-4)

        return skews

    def generate_turbulence_levels(self):
        """
        Generate turbulence levels for all timesteps and groups in one call

        Returns:
        tuple: (turbulence_array, turbulence_gini)
            - turbulence_array: np.array of shape [n_groups, n_timesteps+1] with turbulence levels
            - turbulence_gini: np.array of shape [n_timesteps+1] with Gini coefficients
        """
        # Initialize arrays
        turbulence_array = np.zeros((self.num_groups, self.time_steps + 1))
        turbulence_gini = np.zeros(self.time_steps + 1)

        # Get disaster timesteps
        disaster_timesteps = np.where(self.disaster_impacts > 0)[0]
        # print(f"Found {len(disaster_timesteps)} disaster timesteps")

        for t in disaster_timesteps:
            # Skip if no impact
            if self.disaster_impacts[t] <= 0:
                continue

            # Calculate theoretical turbulence for each group
            # The skew determines how the impact is distributed among groups
            theoretical_turbulence = (
                self.disaster_skews[:, t] * self.disaster_impacts[t] * self.num_groups
            )

            # Allocate and clip turbulence levels properly
            allocated_turbulence = self.allocate_turbulence(theoretical_turbulence)

            # Store in the main array
            turbulence_array[:, t] = allocated_turbulence

            # Calculate and store Gini coefficient if there's any turbulence
            if np.any(allocated_turbulence > 0):
                turbulence_gini[t] = gini_coefficient(allocated_turbulence)

            # Validate allocation - total turbulence / num_groups should approximately equal disaster_impact
            total_allocation = np.sum(allocated_turbulence) / self.num_groups
            expected_allocation = self.disaster_impacts[t]

            # Test that allocation is within reasonable bounds
            tolerance = 0.01  # 1% tolerance
            assert (
                abs(total_allocation - expected_allocation)
                < tolerance * expected_allocation
            ), (
                f"Turbulence allocation error at t={t}: got {total_allocation}, expected {expected_allocation}"
            )

            # print(
            #    f"T={t}: Disaster impact={self.disaster_impacts[t]:.4f}, "
            #    f"Allocated={total_allocation:.4f}, Gini={turbulence_gini[t]:.4f}"
            # )

        # Debug info about the result
        # print(f"Total disaster events: {len(disaster_timesteps)}")
        # print(f"Max turbulence value: {np.max(turbulence_array)}")
        # print(
        #     f"Mean turbulence (when > 0): {np.mean(turbulence_array[turbulence_array > 0]):.4f}"
        # )

        return turbulence_array, turbulence_gini

    def allocate_turbulence(self, theoretical_turbulence):
        """
        Allocate turbulence values to groups, ensuring constraints are met:
        1. No group can have turbulence > 1.0
        2. Total turbulence should be conserved by redistributing excess

        Args:
            theoretical_turbulence: Array of unconstrained turbulence values

        Returns:
            Properly allocated turbulence values
        """
        # Create a copy to avoid modifying the input
        allocated = np.copy(theoretical_turbulence)

        # First, identify groups exceeding max turbulence
        excess_mask = allocated > 1.0

        # If no excess, return as is
        if not np.any(excess_mask):
            return allocated

        # Calculate total excess
        excess_amount = np.sum(allocated[excess_mask] - 1.0)
        # print(
        #     f"Excess turbulence: {excess_amount:.4f} from {np.sum(excess_mask)} groups"
        # )

        # Clip the exceeding groups to 1.0
        allocated[excess_mask] = 1.0

        # Identify groups that can receive the redistributed excess
        can_receive_mask = allocated < 1.0

        # If no groups can receive, we can't redistribute
        if not np.any(can_receive_mask):
            print(
                "Warning: Cannot redistribute excess turbulence - all groups at maximum"
            )
            return allocated

        # Calculate capacity for each group to receive more turbulence
        remaining_capacity = 1.0 - allocated[can_receive_mask]
        total_capacity = np.sum(remaining_capacity)

        # If capacity is less than excess, we'll only redistribute what we can
        redistributable_amount = min(excess_amount, total_capacity)

        if redistributable_amount < excess_amount:
            print(
                f"Warning: Can only redistribute {redistributable_amount:.4f} out of {excess_amount:.4f}"
            )

        # Redistribute proportionally to remaining capacity
        if total_capacity > 0:
            # Calculate fraction of excess to add to each group
            redistribution_fractions = remaining_capacity / total_capacity
            # Add the redistributed amount
            allocated[can_receive_mask] += (
                redistribution_fractions * redistributable_amount
            )

        # Verify and log the final allocation
        clipped_sum = np.sum(allocated)
        original_sum = np.sum(theoretical_turbulence)

        # Check conservation with some tolerance
        conservation_error = (
            abs(clipped_sum - original_sum) / original_sum if original_sum > 0 else 0
        )
        # print(f"Turbulence conservation error: {conservation_error:.4f}")
        if conservation_error > 0.05:  # 5% tolerance
            print("Warning: Significant turbulence loss during redistribution")

        return allocated

    def test_turbulence_allocation(self):
        """
        Test function for turbulence allocation to verify it works correctly
        """
        # Test cases
        test_cases = [
            # Simple case with no clipping needed
            np.array([0.1, 0.2, 0.3, 0.4]),
            # Case with one group exceeding
            np.array([0.5, 1.2, 0.3, 0.2]),
            # Case with multiple groups exceeding
            np.array([1.1, 1.2, 1.3, 0.2]),
            # Extreme case with all groups exceeding
            np.array([1.1, 1.3, 1.5, 1.7]),
        ]

        print("\nRunning turbulence allocation tests:")
        for i, test_input in enumerate(test_cases):
            print(f"\nTest case {i + 1}:")
            print(f"Input: {test_input}")

            result = self.allocate_turbulence(test_input)
            print(f"Result: {result}")

            # Verify constraints
            assert np.all(result <= 1.0), "Constraint violated: Some values exceed 1.0"

            # Check if distribution is sensible
            original_sum = np.sum(test_input)
            result_sum = np.sum(result)

            # Conservation should be close if possible
            if np.any(test_input > 1.0) and np.any(test_input < 1.0):
                # We expect some conservation when redistribution is possible
                conservation_ratio = result_sum / original_sum
                print(f"Conservation ratio: {conservation_ratio:.4f}")
                # We allow some loss if not all excess can be redistributed
                assert conservation_ratio >= 0.95, "Significant turbulence loss"

            print(f"Test case {i + 1} passed!")

        print("\nAll allocation tests passed!")
