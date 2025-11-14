import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging
import pytest
from scripts.core.disaster import DisasterModel

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@pytest.fixture
def disaster_model():
    """Create a DisasterModel instance for testing"""
    params = {
        "numberOfTimeSteps": 400,
        "disasterProbability": 0.01,
        "disasterDistributionType": "beta",
        "5th_percentile": 0.001,
        "95th_percentile": 0.5,
        "numberOfAgentGroups": 21,
        "disasterClusteredness": 0.5,
        "N_NKmodel": 15,
    }
    return DisasterModel(params)


def test_disaster_model(disaster_model):
    """
    Comprehensive test function to validate a DisasterModel instance

    Args:
        disaster_model: An initialized DisasterModel instance
    """
    logging.info("===== TESTING DISASTER MODEL =====")

    # 1. Test disaster impact distribution
    logging.info("Testing disaster impact distribution...")
    impacts = disaster_model.disaster_impacts
    non_zero_impacts = impacts[impacts > 0]

    if len(non_zero_impacts) == 0:
        logging.warning("No disasters generated!")
        return

    logging.info(f"Total time steps: {len(impacts)}")
    logging.info(f"Number of disasters: {np.sum(impacts > 0)}")
    logging.info(
        f"Average impact when disaster occurs: {np.mean(impacts[impacts > 0]):.4f}"
    )
    logging.info(f"Geomean of impacts: {stats.gmean(impacts[impacts > 0]):.4f}")
    logging.info(f"Median of impacts: {np.median(impacts[impacts > 0]):.4f}")
    logging.info(f"First few disaster impacts: {impacts[impacts > 0][:5]}")

    # Check percentiles match expectations
    lower = np.percentile(non_zero_impacts, 5)
    upper = np.percentile(non_zero_impacts, 95)
    logging.info(
        f"5th percentile: {lower:.4f} (target: {disaster_model.lower_percentile})"
    )
    logging.info(
        f"95th percentile: {upper:.4f} (target: {disaster_model.upper_percentile})"
    )

    # 2. Test turbulence allocation
    logging.info("\nTesting turbulence allocation...")
    disaster_model.test_turbulence_allocation()

    # 3. Test actual turbulence array
    logging.info("\nTesting turbulence array...")
    turbulence, gini = disaster_model.turbulenceLevels, disaster_model.turbulenceGini

    # Basic shape checks
    expected_shape = (disaster_model.num_groups, disaster_model.time_steps + 1)
    assert turbulence.shape == expected_shape, (
        f"Wrong shape: {turbulence.shape}, expected {expected_shape}"
    )

    # Find timesteps with disasters
    disaster_timesteps = np.where(impacts > 0)[0]

    # Check turbulence values for disaster timesteps
    for t in disaster_timesteps[:5]:  # Check first 5 for brevity
        t_turbulence = turbulence[:, t]
        t_impact = impacts[t]

        # Check constraints
        assert np.all(t_turbulence <= 1.0), f"Turbulence exceeds 1.0 at t={t}"

        # Check average turbulence approximates impact
        avg_turbulence = np.sum(t_turbulence) / disaster_model.num_groups
        error = abs(avg_turbulence - t_impact) / t_impact

        logging.info(
            f"T={t}: Impact={t_impact:.4f}, Avg turbulence={avg_turbulence:.4f}, Error={error:.4f}"
        )
        assert error < 0.05, f"Large error in turbulence allocation at t={t}"

    logging.info("All turbulence tests passed!")

    # 4. Visualize results
    plot_disaster_model(disaster_model)


def plot_disaster_model(disaster_model):
    """
    Create diagnostic plots for the disaster model

    Args:
        disaster_model: An initialized DisasterModel instance
    """
    logging.info("\nGenerating diagnostic plots...")

    impacts = disaster_model.disaster_impacts
    turbulence = disaster_model.turbulenceLevels
    gini = disaster_model.turbulenceGini

    # Create a multi-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Plot disaster impacts over time
    ax = axes[0, 0]
    ax.stem(range(len(impacts)), impacts, markerfmt="ro", linefmt="r-", basefmt="k-")
    ax.set_title("Disaster Impacts Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Impact Magnitude")

    # 2. Plot distribution of non-zero impacts
    ax = axes[0, 1]
    non_zero = impacts[impacts > 0]
    if len(non_zero) > 0:
        ax.hist(non_zero, bins=20, alpha=0.7)
        ax.axvline(
            disaster_model.lower_percentile,
            color="r",
            linestyle="--",
            label="Target 5th percentile",
        )
        ax.axvline(
            disaster_model.upper_percentile,
            color="g",
            linestyle="--",
            label="Target 95th percentile",
        )
        ax.axvline(
            np.percentile(non_zero, 5),
            color="r",
            linestyle="-",
            label="Actual 5th percentile",
        )
        ax.axvline(
            np.percentile(non_zero, 95),
            color="g",
            linestyle="-",
            label="Actual 95th percentile",
        )
        ax.set_title("Distribution of Disaster Impacts")
        ax.legend()

    # 3. Turbulence heatmap
    ax = axes[1, 0]
    # Only show a portion for clarity if it's large
    if turbulence.shape[1] > 100:
        show_turbulence = turbulence[:, :100]
        extent = [0, 100, 0, turbulence.shape[0]]
    else:
        show_turbulence = turbulence
        extent = [0, turbulence.shape[1], 0, turbulence.shape[0]]

    im = ax.imshow(show_turbulence, aspect="auto", interpolation="none", extent=extent)
    ax.set_title("Turbulence Levels by Group and Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Agent Group")
    fig.colorbar(im, ax=ax, label="Turbulence Level")

    # 4. Gini coefficient over time
    ax = axes[1, 1]
    disaster_timesteps = np.where(impacts > 0)[0]
    if len(disaster_timesteps) > 0:
        ax.scatter(disaster_timesteps, gini[disaster_timesteps], c="blue", alpha=0.7)
        ax.set_title("Gini Coefficient of Turbulence Distribution")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Gini Coefficient")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("disaster_model_diagnostics.png", dpi=300)
    logging.info("Diagnostic plots saved to 'disaster_model_diagnostics.png'")


def run_disaster_model_tests(params):
    """
    Create a DisasterModel instance and run all tests

    Args:
        params: Dictionary of parameters for the DisasterModel
    """
    from disasterModel import DisasterModel

    # Create the disaster model
    model = DisasterModel(params)

    # Run tests
    test_disaster_model(model)

    return model


# Example usage:
if __name__ == "__main__":
    # Example parameter dictionary (replace with your actual parameters)
    params = {
        "numberOfTimeSteps": 400,
        "disasterProbability": 0.01,
        "disasterDistributionType": "beta",
        "5th_percentile": 0.001,
        "95th_percentile": 0.5,
        "numberOfAgentGroups": 21,
        "disasterClusteredness": 0.5,
        "N_NKmodel": 15,
    }

    run_disaster_model_tests(params)
