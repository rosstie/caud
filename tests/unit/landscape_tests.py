import pytest
import numpy as np
from copy import deepcopy
from scripts.core.landscape import FitnessLandscape


def get_test_parameters():
    """Returns parameter dict for testing"""
    p = {}
    p["numberOfAgents"] = 100
    p["numberOfAgentGroups"] = 3
    p["N_NKmodel"] = 15
    p["K_NKmodel"] = 0
    p["scaleNKFitness"] = True
    # Add other required parameters...
    return p


class TestNKLandscapeChange:
    @pytest.fixture
    def setup_basic_landscape(self):
        """Setup basic K=0 landscape for testing"""
        params = get_test_parameters()
        return FitnessLandscape(params)

    @pytest.fixture
    def setup_complex_landscape(self):
        """Setup K=7 landscape for testing"""
        params = get_test_parameters()
        params["K_NKmodel"] = 7
        return FitnessLandscape(params)

    def test_magnitude_monotonicity(self, setup_basic_landscape):
        """Test that larger turbulence leads to larger magnitude changes on average"""
        landscape = setup_basic_landscape
        n_trials = 10  # Increase number of trials for better statistics

        # Test with increasing turbulence levels
        turbulence_levels = [0.1, 0.3, 0.5, 0.7]
        avg_changes = []  # Track changes after power-8 normalization

        for level in turbulence_levels:
            trial_changes = []
            for _ in range(n_trials):
                # Store initial state
                initial_landscape = deepcopy(landscape.landscape)

                # Apply turbulence
                turb = np.ones(landscape.numberOfAgentGroups) * level
                landscape.landscapeChange(turb)

                # Calculate changes after power-8 normalization
                # This is what we actually care about for the rugged landscape
                changes = np.abs(landscape.landscape - initial_landscape)
                avg_change = np.mean(changes)
                trial_changes.append(avg_change)

            avg_changes.append(np.mean(trial_changes))

        # Check that average changes generally increase
        # Allow small decreases (up to 20%) to account for stochasticity
        for i in range(len(avg_changes) - 1):
            ratio = avg_changes[i + 1] / avg_changes[i]
            assert ratio > 0.8, (
                f"Changes decreased too much between {turbulence_levels[i]} and {turbulence_levels[i + 1]}"
            )

        # Also verify that highest turbulence produces largest changes
        assert avg_changes[-1] > avg_changes[0] * 1.5, (
            "Highest turbulence should produce significantly larger changes"
        )

    def test_subcombination_impact(self, setup_basic_landscape):
        """Test that changes affect the entire landscape due to subcombination perturbation"""
        landscape = setup_basic_landscape

        # Store initial state
        initial_landscape = deepcopy(landscape.landscape)

        # Apply moderate turbulence
        turbulence_levels = np.ones(landscape.numberOfAgentGroups) * 0.5
        overall_metrics, group_metrics = landscape.landscapeChange(turbulence_levels)

        # Check that changes are widespread
        for group_idx in range(landscape.numberOfAgentGroups):
            changes = np.abs(
                landscape.landscape[group_idx] - initial_landscape[group_idx]
            )
            # Most positions should show some change due to subcombination perturbation
            fraction_changed = np.mean(changes > 0.01)  # Using small threshold
            assert fraction_changed > 0.5, (
                "Expected widespread changes due to subcombination perturbation"
            )

    def test_interaction_matrix_preservation(self, setup_complex_landscape):
        """Test interaction matrices remain unchanged through disaster"""
        landscape = setup_complex_landscape

        # Store initial matrices
        initial_matrices = deepcopy(landscape.interactionMatrices)

        # Apply disaster with uniform turbulence
        turbulence_levels = (
            np.ones(landscape.numberOfAgentGroups) / landscape.numberOfAgentGroups
        )
        overall_metrics, group_metrics = landscape.landscapeChange(turbulence_levels)

        # Verify matrices unchanged
        for i, matrix in enumerate(landscape.interactionMatrices):
            assert np.array_equal(matrix, initial_matrices[i])

    def test_max_payoff_normalization(self, setup_basic_landscape):
        """Test landscape remains normalized after changes"""
        landscape = setup_basic_landscape

        # Check initial normalization
        for group in range(landscape.numberOfAgentGroups):
            assert np.max(landscape.landscape[group]) == pytest.approx(1.0)

        # Apply disaster with uniform turbulence
        turbulence_levels = (
            np.ones(landscape.numberOfAgentGroups) / landscape.numberOfAgentGroups
        )
        overall_metrics, group_metrics = landscape.landscapeChange(turbulence_levels)

        # Check post-disaster normalization
        for group in range(landscape.numberOfAgentGroups):
            assert np.max(landscape.landscape[group]) == pytest.approx(1.0)

    def test_k0_vs_k7_complexity(self, setup_basic_landscape, setup_complex_landscape):
        """Test that K=7 landscapes show more complex changes than K=0 on average"""
        k0_landscape = setup_basic_landscape
        k7_landscape = setup_complex_landscape
        n_trials = 10  # Increase number of trials
        turbulence_level = 0.3

        k0_complexity = []
        k7_complexity = []

        for _ in range(n_trials):
            # Store initial states
            k0_initial = deepcopy(k0_landscape.landscape)
            k7_initial = deepcopy(k7_landscape.landscape)

            # Apply same turbulence to both
            turbulence_levels = (
                np.ones(k0_landscape.numberOfAgentGroups) * turbulence_level
            )

            # Apply changes
            k0_landscape.landscapeChange(turbulence_levels)
            k7_landscape.landscapeChange(turbulence_levels)

            # Calculate complexity metrics
            def calc_complexity(old, new):
                # Calculate changes after power-8 normalization
                changes = np.abs(new - old)

                # Calculate spatial correlation of changes
                spatial_corr = np.mean(
                    [
                        np.corrcoef(changes[i], np.roll(changes[i], 1))[0, 1]
                        for i in range(len(changes))
                    ]
                )

                # Calculate RMSD using the new landscape mean for normalization
                # This better reflects the actual change in landscape structure
                rmsd = np.sqrt(np.mean((new - old) ** 2)) / np.mean(new)

                # Combine both metrics - we want low spatial correlation and high RMSD
                return (1 - spatial_corr) * rmsd

            k0_complexity.append(calc_complexity(k0_initial, k0_landscape.landscape))
            k7_complexity.append(calc_complexity(k7_initial, k7_landscape.landscape))

        # Compare average complexity
        avg_k0_complexity = np.mean(k0_complexity)
        avg_k7_complexity = np.mean(k7_complexity)
        print(f"Average K0 complexity: {avg_k0_complexity}")
        print(f"Average K7 complexity: {avg_k7_complexity}")

        # K7 should show more complex changes (lower spatial correlation and higher RMSD)
        assert avg_k7_complexity > avg_k0_complexity * 1.2, (
            "K=7 landscape should show significantly more complex changes than K=0"
        )

    def test_correlation_preservation(self, setup_complex_landscape):
        """Test that landscape changes preserve some correlation with original landscape"""
        landscape = setup_complex_landscape

        # Store initial state
        initial_landscape = deepcopy(landscape.landscape)

        # Apply moderate turbulence
        turbulence_levels = np.ones(landscape.numberOfAgentGroups) * 0.3
        overall_metrics, group_metrics = landscape.landscapeChange(turbulence_levels)

        # Check correlations
        for group_idx in range(landscape.numberOfAgentGroups):
            correlation = np.corrcoef(
                initial_landscape[group_idx], landscape.landscape[group_idx]
            )[0, 1]
            # Should maintain some correlation with original landscape
            assert correlation > 0.0, (
                "Changed landscape should maintain some correlation with original"
            )
            # But shouldn't be perfectly correlated
            assert correlation < 0.99, (
                "Changed landscape shouldn't be perfectly correlated with original"
            )


if __name__ == "__main__":
    pytest.main(["-v", "landscape_tests.py"])
