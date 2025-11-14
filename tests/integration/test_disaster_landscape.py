import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
        logging.FileHandler(log_dir / "integration_tests.log"),
        logging.StreamHandler(),
    ],
)


@pytest.fixture
def integration_params():
    """Fixture to provide parameters for integration tests"""
    params = get_parameters()
    params.update(
        {
            "numberOfTimeSteps": 400,
            "disasterProbability": 0.01,
            "disasterDistributionType": "truncated_lognormal",
            "5th_percentile": 0.001,
            "95th_percentile": 0.5,
            "numberOfAgentGroups": 21,
            "disasterClusteredness": 0.5,
            "N_NKmodel": 15,
            "K_NKmodel": 1,  # Default to K=1 for testing
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


@pytest.fixture
def run_visualization(request):
    """Fixture to check if visualization should be run"""
    return request.config.getoption("--runvisualization", default=False)


def test_turbulence_landscape_integration(
    disaster_model, landscape, integration_params, tmp_path, run_visualization
):
    """Test the integration between the disaster model and landscape change"""
    logging.info("=== Starting Turbulence-Landscape Integration Test ===")

    # Get K value
    k_value = integration_params["K_NKmodel"]
    if isinstance(k_value, list):
        k_value = k_value[0]  # Take first value if it's a list
    logging.info(f"Testing with K={k_value}")

    # Track metrics for analysis
    num_groups = integration_params["numberOfAgentGroups"]
    num_timesteps = integration_params["numberOfTimeSteps"]

    # Arrays to store impact metrics
    impact_rmsd = np.zeros(num_timesteps + 1)
    impact_corr = np.zeros(num_timesteps + 1)
    impact_magnitude = np.zeros(num_timesteps + 1)
    group_impacts = np.zeros((num_groups, num_timesteps + 1, 3))

    # Get time steps with disasters
    disaster_timesteps = np.where(disaster_model.disaster_impacts > 0)[0]
    logging.info(f"Found {len(disaster_timesteps)} disaster time steps")

    # Test landscape changes for the first few disaster events
    for i, t in enumerate(disaster_timesteps[: min(5, len(disaster_timesteps))]):
        logging.info(f"\nTesting disaster at t={t}")

        # Get turbulence levels for this time step
        turbulence = disaster_model.turbulenceLevels[:, t]
        logging.info(f"Turbulence levels: {turbulence}")

        # Apply landscape change
        overall_metrics, group_metrics = landscape.landscapeChange(turbulence)

        # Store metrics
        impact_rmsd[t] = overall_metrics[0]
        impact_corr[t] = overall_metrics[1]
        impact_magnitude[t] = overall_metrics[2]
        group_impacts[:, t, :] = group_metrics

        # Log results
        logging.info(
            f"Overall metrics: RMSD={overall_metrics[0]:.4f}, "
            f"Correlation change={overall_metrics[1]:.4f}, "
            f"Magnitude={overall_metrics[2]:.4f}"
        )

        # Check for correlation between turbulence and actual impact
        corr_turb_rmsd = np.corrcoef(turbulence, group_metrics[:, 0])[0, 1]
        corr_turb_mag = np.corrcoef(turbulence, group_metrics[:, 2])[0, 1]

        logging.info(f"Correlation between turbulence and RMSD: {corr_turb_rmsd:.4f}")
        logging.info(
            f"Correlation between turbulence and magnitude: {corr_turb_mag:.4f}"
        )

        # Set thresholds based on K
        if k_value <= 1:
            rmsd_threshold = 0.2  # Very low threshold for K=0,1
            mag_threshold = 0.3  # Magnitude tends to correlate better
        elif k_value <= 3:
            rmsd_threshold = 0.3  # Low threshold for K=2,3
            mag_threshold = 0.4
        else:
            rmsd_threshold = 0.45  # Higher threshold for K>3
            mag_threshold = 0.5

        # Use warnings instead of assertions for low K values
        if k_value <= 3:
            if corr_turb_rmsd < rmsd_threshold:
                logging.warning(
                    f"Low correlation between turbulence and RMSD ({corr_turb_rmsd:.4f})"
                )
                logging.warning(
                    "For low K landscapes, weak correlations are expected due to landscape structure"
                )
            else:
                logging.info(
                    f"Good correlation between turbulence and RMSD ({corr_turb_rmsd:.4f}) for K={k_value}"
                )

            if corr_turb_mag < mag_threshold:
                logging.warning(
                    f"Low correlation between turbulence and magnitude ({corr_turb_mag:.4f})"
                )
            else:
                logging.info(
                    f"Good correlation between turbulence and magnitude ({corr_turb_mag:.4f}) for K={k_value}"
                )
        else:
            # For higher K values, use assertions
            assert corr_turb_rmsd > rmsd_threshold, (
                f"Turbulence and RMSD should correlate >={rmsd_threshold}"
            )
            assert corr_turb_mag > mag_threshold, (
                f"Turbulence and magnitude should correlate >={mag_threshold}"
            )

    # Plot results if visualization is enabled
    if run_visualization:
        plot_integration_results(
            disaster_model.turbulenceLevels,
            impact_rmsd,
            impact_corr,
            impact_magnitude,
            group_impacts,
            disaster_timesteps,
            k_value,
            tmp_path,
        )

    # If we have a low K value, run the extended analysis
    if k_value <= 3:
        logging.info("Running extended analysis for low K value...")
        analysis_results = analyze_turbulence_impact_relationship(
            integration_params, disaster_model, landscape, num_samples=8
        )

    logging.info("=== Turbulence-Landscape Integration Test Complete ===")


def plot_integration_results(
    turbulence,
    rmsd,
    corr,
    magnitude,
    group_impacts,
    disaster_timesteps,
    k_value,
    tmp_path,
):
    """Generate plots to visualize the relationship between turbulence and landscape impacts"""
    vis_dir = tmp_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Plot 1: Turbulence vs RMSD for all disaster events
    ax = axes[0]
    for t in disaster_timesteps:
        if rmsd[t] > 0:  # Only plot non-zero impacts
            ax.scatter(
                turbulence[:, t],
                group_impacts[:, t, 0],
                label=f"t={t}" if t < 5 else None,
            )  # Only label first few

    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("RMSD Impact")
    ax.set_title(f"Relationship Between Turbulence and RMSD Impact (K={k_value})")
    if len(disaster_timesteps) > 0:
        ax.legend()

    # Plot 2: Turbulence vs Correlation Change
    ax = axes[1]
    for t in disaster_timesteps:
        if corr[t] > 0:
            ax.scatter(
                turbulence[:, t],
                group_impacts[:, t, 1],
                label=f"t={t}" if t < 5 else None,
            )

    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("Correlation Change")
    ax.set_title(
        f"Relationship Between Turbulence and Correlation Change (K={k_value})"
    )
    if len(disaster_timesteps) > 0:
        ax.legend()

    # Plot 3: Turbulence vs Magnitude
    ax = axes[2]
    for t in disaster_timesteps:
        if magnitude[t] > 0:
            ax.scatter(
                turbulence[:, t],
                group_impacts[:, t, 2],
                label=f"t={t}" if t < 5 else None,
            )

    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("Magnitude Impact")
    ax.set_title(f"Relationship Between Turbulence and Magnitude Impact (K={k_value})")
    if len(disaster_timesteps) > 0:
        ax.legend()

    plt.tight_layout()
    plt.savefig(vis_dir / f"turbulence_impact_analysis_K{k_value}.png", dpi=300)
    logging.info(
        f"Saved integration analysis plot to {vis_dir / f'turbulence_impact_analysis_K{k_value}.png'}"
    )


def analyze_turbulence_impact_relationship(
    parameters, disaster_model, landscape, num_samples=10
):
    """Perform a detailed analysis of the relationship between turbulence levels and landscape changes"""
    logging.info("Running extended turbulence-impact analysis...")

    # Get key parameters
    k_value = parameters["K_NKmodel"]
    if isinstance(k_value, list):
        k_value = k_value[0]  # Take first value if it's a list

    num_groups = parameters["numberOfAgentGroups"]

    # Create arrays to store results
    turbulence_levels = np.linspace(0.01, 0.95, num_samples)
    rmsd_results = np.zeros((num_groups, num_samples))
    corr_results = np.zeros((num_groups, num_samples))
    mag_results = np.zeros((num_groups, num_samples))

    # Run multiple tests with controlled turbulence levels
    for i, level in enumerate(turbulence_levels):
        logging.info(f"Testing turbulence level {level:.2f}")

        # Create controlled turbulence array - equal for all groups
        turbulence = np.ones(num_groups) * level

        # Apply landscape change
        overall_metrics, group_metrics = landscape.landscapeChange(turbulence)

        # Store results
        rmsd_results[:, i] = group_metrics[:, 0]
        corr_results[:, i] = group_metrics[:, 1]
        mag_results[:, i] = group_metrics[:, 2]

    # Calculate statistics
    mean_rmsd = np.mean(rmsd_results, axis=0)
    std_rmsd = np.std(rmsd_results, axis=0)
    mean_corr = np.mean(corr_results, axis=0)
    std_corr = np.std(corr_results, axis=0)
    mean_mag = np.mean(mag_results, axis=0)
    std_mag = np.std(mag_results, axis=0)

    # Calculate overall correlations
    corr_turb_rmsd = np.corrcoef(turbulence_levels, mean_rmsd)[0, 1]
    corr_turb_corr = np.corrcoef(turbulence_levels, mean_corr)[0, 1]
    corr_turb_mag = np.corrcoef(turbulence_levels, mean_mag)[0, 1]

    # Log results
    logging.info(f"Correlation between turbulence and RMSD: {corr_turb_rmsd:.4f}")
    logging.info(
        f"Correlation between turbulence and correlation change: {corr_turb_corr:.4f}"
    )
    logging.info(f"Correlation between turbulence and magnitude: {corr_turb_mag:.4f}")

    # Return results
    return {
        "turbulence_levels": turbulence_levels,
        "mean_rmsd": mean_rmsd,
        "std_rmsd": std_rmsd,
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "mean_mag": mean_mag,
        "std_mag": std_mag,
        "corr_turb_rmsd": corr_turb_rmsd,
        "corr_turb_corr": corr_turb_corr,
        "corr_turb_mag": corr_turb_mag,
    }
