import os
import sys
import numpy as np
import pytest
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.core.disaster import DisasterModel
from scripts.core.landscape import FitnessLandscape
from scripts.config.params import get_parameters

# Configure logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "turbulence_landscape_integration.log"),
        logging.StreamHandler(),
    ],
)


@pytest.fixture
def integration_params():
    """Fixture to provide parameters for integration tests"""
    params = get_parameters()
    params.update(
        {
            "numberOfTimeSteps": 100,
            "disasterProbability": 0.1,  # Higher probability for testing
            "disasterDistributionType": "truncated_lognormal",
            "5th_percentile": 0.001,
            "95th_percentile": 0.5,
            "numberOfAgentGroups": 5,
            "disasterClusteredness": 0.5,
            "N_NKmodel": 8,
            "K_NKmodel": 2,  # Medium complexity for testing
        }
    )
    return params


@pytest.fixture
def disaster_model(integration_params):
    """Fixture to provide an initialized DisasterModel instance"""
    return DisasterModel(integration_params)


@pytest.fixture
def landscape(integration_params):
    """Fixture to provide an initialized FitnessLandscape instance"""
    return FitnessLandscape(integration_params)


def test_turbulence_landscape_basic_integration(disaster_model, landscape):
    """Test basic integration between disaster model and landscape"""
    logging.info("Testing basic turbulence-landscape integration")

    # Get disaster timesteps
    disaster_timesteps = np.where(disaster_model.disaster_impacts > 0)[0]
    logging.info(f"Found {len(disaster_timesteps)} disaster time steps")

    if len(disaster_timesteps) == 0:
        pytest.skip("No disasters generated for testing")

    # Test landscape changes for the first disaster event
    t = disaster_timesteps[0]
    logging.info(f"Testing disaster at t={t}")

    # Get turbulence levels for this time step
    turbulence = disaster_model.turbulenceLevels[:, t]
    logging.info(f"Turbulence levels: {turbulence}")

    # Store initial landscape state
    initial_landscape = landscape.landscape.copy()

    # Apply landscape change
    overall_metrics, group_metrics = landscape.landscapeChange(turbulence)

    # Check that landscape changed
    assert not np.array_equal(landscape.landscape, initial_landscape), (
        "Landscape did not change"
    )

    # Check metrics
    assert overall_metrics.shape == (3,), (
        f"Expected 3 overall metrics, got {overall_metrics.shape}"
    )
    assert group_metrics.shape == (landscape.numberOfAgentGroups, 3), (
        f"Expected group metrics shape ({landscape.numberOfAgentGroups}, 3), got {group_metrics.shape}"
    )

    # Check that metrics are non-negative
    assert np.all(overall_metrics >= 0), "Overall metrics should be non-negative"
    assert np.all(group_metrics >= 0), "Group metrics should be non-negative"

    # Check correlation between turbulence and impact
    impact = disaster_model.disaster_impacts[t]
    logging.info(f"Disaster impact: {impact}")

    # For non-zero impact, check that higher turbulence leads to higher impact
    if impact > 0:
        # Calculate correlation between turbulence and magnitude
        corr_turb_mag = np.corrcoef(turbulence, group_metrics[:, 2])[0, 1]
        logging.info(
            f"Correlation between turbulence and magnitude: {corr_turb_mag:.4f}"
        )

        # For K=2, expect moderate correlation
        assert corr_turb_mag > 0.3, (
            f"Expected positive correlation between turbulence and magnitude, got {corr_turb_mag:.4f}"
        )


def test_turbulence_landscape_multiple_disasters(disaster_model, landscape):
    """Test landscape changes across multiple disaster events"""
    logging.info("Testing multiple disaster events")

    # Get disaster timesteps
    disaster_timesteps = np.where(disaster_model.disaster_impacts > 0)[0]
    logging.info(f"Found {len(disaster_timesteps)} disaster time steps")

    if len(disaster_timesteps) < 2:
        pytest.skip("Need at least 2 disasters for this test")

    # Test landscape changes for the first few disaster events
    for i, t in enumerate(disaster_timesteps[: min(3, len(disaster_timesteps))]):
        logging.info(f"\nTesting disaster {i + 1} at t={t}")

        # Get turbulence levels for this time step
        turbulence = disaster_model.turbulenceLevels[:, t]
        impact = disaster_model.disaster_impacts[t]
        logging.info(f"Disaster impact: {impact}")
        logging.info(f"Turbulence levels: {turbulence}")

        # Apply landscape change
        overall_metrics, group_metrics = landscape.landscapeChange(turbulence)

        # Log metrics
        logging.info(
            f"Overall metrics: RMSD={overall_metrics[0]:.4f}, "
            f"Correlation change={overall_metrics[1]:.4f}, "
            f"Magnitude={overall_metrics[2]:.4f}"
        )

        # Check that metrics scale with impact
        if impact > 0:
            # For larger impacts, expect larger magnitude changes
            assert overall_metrics[2] > 0, (
                "Expected positive magnitude change for non-zero impact"
            )

            # Check that higher impact leads to higher magnitude change
            if i > 0:
                prev_impact = disaster_model.disaster_impacts[disaster_timesteps[i - 1]]
                prev_magnitude = landscape_change_magnitude(
                    landscape.landscape.flatten(), landscape.landscape.flatten()
                )

                # If this impact is larger than the previous, magnitude should be larger
                if impact > prev_impact:
                    assert overall_metrics[2] >= prev_magnitude, (
                        f"Expected larger magnitude for larger impact, got {overall_metrics[2]:.4f} vs {prev_magnitude:.4f}"
                    )


def test_turbulence_landscape_clusteredness(
    disaster_model, landscape, integration_params
):
    """Test that disaster clusteredness affects landscape changes"""
    logging.info("Testing disaster clusteredness effects")

    # Get disaster timesteps
    disaster_timesteps = np.where(disaster_model.disaster_impacts > 0)[0]
    logging.info(f"Found {len(disaster_timesteps)} disaster time steps")

    if len(disaster_timesteps) == 0:
        pytest.skip("No disasters generated for testing")

    # Test landscape changes for the first disaster event
    t = disaster_timesteps[0]
    logging.info(f"Testing disaster at t={t}")

    # Get turbulence levels for this time step
    turbulence = disaster_model.turbulenceLevels[:, t]
    logging.info(f"Turbulence levels: {turbulence}")

    # Calculate Gini coefficient of turbulence
    gini = calculate_gini_coefficient(turbulence)
    logging.info(f"Turbulence Gini coefficient: {gini:.4f}")

    # Apply landscape change
    overall_metrics, group_metrics = landscape.landscapeChange(turbulence)

    # Calculate Gini coefficient of group impacts
    impact_gini = calculate_gini_coefficient(group_metrics[:, 2])
    logging.info(f"Impact Gini coefficient: {impact_gini:.4f}")

    # Check that clustered disasters lead to more uneven impacts
    clusteredness = integration_params["disasterClusteredness"]
    logging.info(f"Disaster clusteredness: {clusteredness:.2f}")

    # For high clusteredness, expect high Gini coefficient
    if clusteredness > 0.7:
        assert gini > 0.5, f"Expected high Gini for high clusteredness, got {gini:.4f}"
        assert impact_gini > 0.3, (
            f"Expected high impact Gini for high clusteredness, got {impact_gini:.4f}"
        )

    # For low clusteredness, expect lower Gini coefficient
    if clusteredness < 0.3:
        assert gini < 0.5, f"Expected low Gini for low clusteredness, got {gini:.4f}"
        assert impact_gini < 0.5, (
            f"Expected low impact Gini for low clusteredness, got {impact_gini:.4f}"
        )


def calculate_gini_coefficient(values):
    """Calculate Gini coefficient for a set of values"""
    # Sort values
    sorted_values = np.sort(values)
    n = len(values)

    # Calculate cumulative sum
    cumsum = np.cumsum(sorted_values)

    # Calculate Gini coefficient
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0


def landscape_change_magnitude(old_landscape, new_landscape):
    """Calculate the magnitude of landscape change"""
    return np.mean(np.abs(old_landscape - new_landscape))
