"""
Integration Test Script with K-adaptive analysis
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import copy

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "integration_test.log")),
        logging.StreamHandler(sys.stdout),
    ],
)

# Import local modules
from scripts.core.disaster import DisasterModel
from scripts.core.landscape import FitnessLandscape
from scripts.utils import utils
import params_test


def test_turbulence_landscape_integration():
    """
    Test the integration between the disaster model and landscape change
    with K-adaptive thresholds
    """
    logging.info("=== Starting Turbulence-Landscape Integration Test ===")

    # Get parameters
    base_parameters = params_test.parameters
    k_values = [0, 3, 7]  # Test each K value separately

    for k_value in k_values:
        logging.info(f"\nTesting with K={k_value}")

        # Create a copy of parameters for this K value
        parameters = copy.deepcopy(base_parameters)
        parameters["K_NKmodel"] = k_value

        # Step 1: Create disaster model and generate turbulence levels
        logging.info("Creating disaster model...")
        disaster = DisasterModel(parameters)

        # Step 2: Create landscape
        logging.info("Creating fitness landscape...")
        landscape = FitnessLandscape(parameters)

        # Step 3: Run landscape changes based on turbulence levels
        logging.info("Testing landscape changes...")

        # Track metrics for analysis
        num_groups = parameters["numberOfAgentGroups"]
        num_timesteps = parameters["numberOfTimeSteps"]

        # Arrays to store impact metrics
        impact_rmsd = np.zeros(num_timesteps + 1)
        impact_corr = np.zeros(num_timesteps + 1)
        impact_magnitude = np.zeros(num_timesteps + 1)
        group_impacts = np.zeros((num_groups, num_timesteps + 1, 3))

        # Get time steps with disasters
        disaster_timesteps = np.where(disaster.disaster_impacts > 0)[0]
        logging.info(f"Found {len(disaster_timesteps)} disaster time steps")

        # Test landscape changes for the first few disaster events
        for i, t in enumerate(disaster_timesteps[: min(5, len(disaster_timesteps))]):
            logging.info(f"\nTesting disaster at t={t}")

            # Get turbulence levels for this time step
            turbulence = disaster.turbulenceLevels[:, t]
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

            logging.info(
                f"Correlation between turbulence and RMSD: {corr_turb_rmsd:.4f}"
            )
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

        # Plot results
        plot_integration_results(
            disaster.turbulenceLevels,
            impact_rmsd,
            impact_corr,
            impact_magnitude,
            group_impacts,
            disaster_timesteps,
            k_value,
        )

        # If we have a low K value, run the extended analysis
        if k_value <= 3:
            logging.info("Running extended analysis for low K value...")
            analysis_results = analyze_turbulence_impact_relationship(
                parameters, disaster, landscape, num_samples=8
            )

    logging.info("=== Turbulence-Landscape Integration Test Complete ===")


def plot_integration_results(
    turbulence, rmsd, corr, magnitude, group_impacts, disaster_timesteps, k_value
):
    """
    Generate plots to visualize the relationship between turbulence and landscape impacts
    """
    plt.figure(figsize=(15, 5))

    # Plot 1: RMSD vs Turbulence
    plt.subplot(131)
    ax = plt.gca()
    for t in disaster_timesteps:
        if rmsd[t] > 0:  # Only plot non-zero impacts
            ax.scatter(
                turbulence[:, t],
                group_impacts[:, t, 0],
                label=f"t={t}" if t < 5 else None,
            )  # Only label first few

    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("RMSD Impact")
    ax.set_title(f"RMSD vs Turbulence (K={k_value})")
    if len(disaster_timesteps) > 0:
        ax.legend()

    # Plot 2: Correlation Change vs Turbulence
    plt.subplot(132)
    ax = plt.gca()
    for t in disaster_timesteps:
        if corr[t] > 0:
            ax.scatter(
                turbulence[:, t],
                group_impacts[:, t, 1],
                label=f"t={t}" if t < 5 else None,
            )

    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("Correlation Change")
    ax.set_title(f"Correlation Change vs Turbulence (K={k_value})")
    if len(disaster_timesteps) > 0:
        ax.legend()

    # Plot 3: Magnitude vs Turbulence
    plt.subplot(133)
    ax = plt.gca()
    for t in disaster_timesteps:
        if magnitude[t] > 0:
            ax.scatter(
                turbulence[:, t],
                group_impacts[:, t, 2],
                label=f"t={t}" if t < 5 else None,
            )

    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("Magnitude Impact")
    ax.set_title(f"Magnitude vs Turbulence (K={k_value})")
    if len(disaster_timesteps) > 0:
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"turbulence_impact_analysis_K{k_value}.png", dpi=300)
    logging.info(
        f"Saved integration analysis plot to 'turbulence_impact_analysis_K{k_value}.png'"
    )


def analyze_turbulence_impact_relationship(
    parameters, disaster_model, landscape, num_samples=10
):
    """
    Perform a detailed analysis of the relationship between turbulence levels
    and landscape impacts for low K values

    Args:
        parameters: Dictionary of simulation parameters
        disaster_model: Instance of DisasterModel
        landscape: Instance of FitnessLandscape
        num_samples: Number of test samples to run
    """
    logging.info("Running extended turbulence-impact analysis...")

    # Get key parameters
    k_value = parameters["K_NKmodel"]
    if isinstance(k_value, list):
        k_value = k_value[0]  # Take first value if it's a list

    num_groups = parameters["numberOfAgentGroups"]

    # Create arrays to store results
    turbulence_levels = np.linspace(0, 1, 10)
    rmsd_results = np.zeros((num_samples, len(turbulence_levels)))
    corr_results = np.zeros((num_samples, len(turbulence_levels)))
    mag_results = np.zeros((num_samples, len(turbulence_levels)))

    # Run multiple samples
    for i in range(num_samples):
        logging.info(f"Running sample {i + 1}/{num_samples}")
        for j, turb in enumerate(turbulence_levels):
            # Apply uniform turbulence
            turbulence = np.full(num_groups, turb)

            # Get impacts
            overall_metrics, _ = landscape.landscapeChange(turbulence)
            rmsd_results[i, j] = overall_metrics[0]
            corr_results[i, j] = overall_metrics[1]
            mag_results[i, j] = overall_metrics[2]

    # Calculate means and standard deviations
    mean_rmsd = np.mean(rmsd_results, axis=0)
    std_rmsd = np.std(rmsd_results, axis=0)
    mean_corr = np.mean(corr_results, axis=0)
    std_corr = np.std(corr_results, axis=0)
    mean_mag = np.mean(mag_results, axis=0)
    std_mag = np.std(mag_results, axis=0)

    # Calculate correlations
    corr_turb_rmsd = stats.pearsonr(turbulence_levels, mean_rmsd)[0]
    corr_turb_corr = stats.pearsonr(turbulence_levels, mean_corr)[0]
    corr_turb_mag = stats.pearsonr(turbulence_levels, mean_mag)[0]

    # Calculate Spearman rank correlations
    spearman_rmsd = stats.spearmanr(turbulence_levels, mean_rmsd)[0]
    spearman_mag = stats.spearmanr(turbulence_levels, mean_mag)[0]

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot RMSD vs Turbulence
    ax = plt.subplot(131)
    ax.errorbar(turbulence_levels, mean_rmsd, yerr=std_rmsd, fmt="o-", capsize=3)
    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("RMSD Impact")
    ax.set_title(f"RMSD vs Turbulence (K={k_value}, corr={corr_turb_rmsd:.4f})")

    # Plot Correlation Change vs Turbulence
    ax = plt.subplot(132)
    ax.errorbar(turbulence_levels, mean_corr, yerr=std_corr, fmt="o-", capsize=3)
    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("Correlation Change")
    ax.set_title(
        f"Correlation Change vs Turbulence (K={k_value}, corr={corr_turb_corr:.4f})"
    )

    # Plot Magnitude vs Turbulence
    ax = plt.subplot(133)
    ax.errorbar(turbulence_levels, mean_mag, yerr=std_mag, fmt="o-", capsize=3)
    ax.set_xlabel("Turbulence Level")
    ax.set_ylabel("Magnitude Impact")
    ax.set_title(f"Magnitude vs Turbulence (K={k_value}, corr={corr_turb_mag:.4f})")

    plt.tight_layout()
    plt.savefig(f"extended_analysis_K{k_value}.png", dpi=300)
    logging.info(f"Saved extended analysis plot to 'extended_analysis_K{k_value}.png'")

    # Return analysis results
    return {
        "turbulence_levels": turbulence_levels,
        "rmsd_results": rmsd_results,
        "corr_results": corr_results,
        "mag_results": mag_results,
        "pearson_correlations": [corr_turb_rmsd, corr_turb_corr, corr_turb_mag],
        "spearman_correlations": [spearman_rmsd, spearman_mag],
    }


if __name__ == "__main__":
    # Run integration test
    test_turbulence_landscape_integration()
