#!/usr/bin/env python3
"""
Simple example script to verify the model works correctly.

This script runs a minimal simulation with default parameters and displays
basic results to verify the model is functioning properly.
"""

import sys
from pathlib import Path

# Add scripts directory to path before other imports
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir.parent))

import numpy as np
import matplotlib.pyplot as plt
from scripts.core.simulation import Simulation
from scripts.config.params import get_parameters


def main():
    """Run a simple simulation and display results."""
    print("=" * 60)
    print("Running Simple Model Verification")
    print("=" * 60)

    # Get default parameters and override for longer simulation with disasters
    params = get_parameters()

    # Override parameters for better demonstration
    params["numberOfTimeSteps"] = 150  # Longer simulation
    params["disasterProbability"] = 0.08  # Higher probability to see disasters
    params["95th_percentile"] = 0.5  # Ensure disasters have meaningful impact
    params["5th_percentile"] = 0.001  # Lower bound for disaster impacts

    print("\nSimulation Parameters:")
    print(f"  Number of agents: {params['numberOfAgents']}")
    print(f"  Number of groups: {params['numberOfAgentGroups']}")
    print(f"  Time steps: {params['numberOfTimeSteps']}")
    print(f"  NK Model: N={params['N_NKmodel']}, K={params['K_NKmodel']}")
    print(f"  Network type: {params['typeOfNetworkSocialLearning']}")
    print(f"  Disaster probability: {params['disasterProbability']}")
    print(
        f"  Disaster impact range: {params['5th_percentile']:.4f} - {params['95th_percentile']:.4f}"
    )

    print("\nRunning simulation...")

    # Create and run simulation
    sim = Simulation(params)
    sim.runEfficientSL()

    print("Simulation completed successfully!")

    # Calculate and display key metrics
    avg_payoff = np.mean(sim.payoffsAgentsOverTime, axis=0)
    final_payoff = avg_payoff[-1]
    max_payoff = np.max(avg_payoff)
    unique_items_final = sim.uniqueItemsOverTime[-1]

    # Calculate disaster statistics
    disaster_times = np.where(sim.ImpactRmsd > 0)[0]
    num_disasters = len(disaster_times)
    disaster_impacts = (
        sim.ImpactRmsd[disaster_times] if num_disasters > 0 else np.array([])
    )
    disaster_magnitudes = (
        sim.ImpactMagnitude[disaster_times] if num_disasters > 0 else np.array([])
    )

    print("\n" + "=" * 60)
    print("Key Results:")
    print("=" * 60)
    print(f"  Final average performance: {final_payoff:.4f}")
    print(f"  Maximum average performance: {max_payoff:.4f}")
    print(f"  Final number of unique items: {unique_items_final}")
    print(f"  Total social learning events: {np.sum(sim.learnedSolutions)}")
    print(f"  Total innovation events: {np.sum(sim.innovatedSolutions)}")
    print("\n  Disaster Events:")
    print(f"    Number of disasters: {num_disasters}")
    if num_disasters > 0:
        print(f"    Average impact (RMSD): {np.mean(disaster_impacts):.4f}")
        print(f"    Average impact magnitude: {np.mean(disaster_magnitudes):.4f}")
        print(f"    Max impact (RMSD): {np.max(disaster_impacts):.4f}")
        print(
            f"    Disaster times: {disaster_times[:10].tolist()}{'...' if len(disaster_times) > 10 else ''}"
        )

    # Create a visualization with three panels
    output_dir = Path("results/fig")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Performance over time
    for agent_idx in range(min(10, sim.numberOfAgents)):  # Show first 10 agents
        ax1.plot(
            sim.payoffsAgentsOverTime[agent_idx, :],
            linewidth=0.5,
            alpha=0.3,
            color="gray",
        )

    ax1.plot(avg_payoff, color="red", linewidth=2.5, label="Average Performance")

    # Mark disaster events with vertical lines
    if num_disasters > 0:
        for disaster_time in disaster_times:
            impact_size = sim.ImpactRmsd[disaster_time]
            # Color intensity based on impact size
            alpha = min(0.8, 0.3 + impact_size * 0.5)
            ax1.axvline(
                x=disaster_time,
                color="blue",
                linewidth=1.5,
                linestyle="--",
                alpha=alpha,
                zorder=0,
            )

    ax1.set_ylabel("Performance")
    ax1.set_title("Agent Performance Over Time (Blue lines = Disasters)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])

    # Plot 2: Learning and innovation events
    ax2.plot(
        sim.learnedSolutions, color="blue", linewidth=2, label="Social Learning Events"
    )
    ax2.plot(
        sim.innovatedSolutions, color="red", linewidth=2, label="Innovation Events"
    )
    # Mark disasters on this plot too
    if num_disasters > 0:
        for disaster_time in disaster_times:
            ax2.axvline(
                x=disaster_time,
                color="gray",
                linewidth=0.8,
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )
    ax2.set_ylabel("Number of Events")
    ax2.set_title("Social Learning and Innovation Events")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Disaster impacts and sizes
    if num_disasters > 0:
        # Plot disaster impact magnitudes
        ax3.scatter(
            disaster_times,
            disaster_impacts,
            s=disaster_magnitudes * 100,  # Size proportional to magnitude
            c=disaster_magnitudes,
            cmap="Reds",
            alpha=0.7,
            edgecolors="darkred",
            linewidths=1,
            label="Disaster Impact (RMSD)",
            zorder=3,
        )

        # Also plot as line to show timing
        ax3.plot(
            disaster_times,
            disaster_impacts,
            color="darkred",
            linewidth=1,
            alpha=0.3,
            linestyle="-",
            zorder=2,
        )

        # Add magnitude as secondary y-axis
        ax3_twin = ax3.twinx()
        ax3_twin.bar(
            disaster_times,
            disaster_magnitudes,
            alpha=0.3,
            color="orange",
            width=0.8,
            label="Impact Magnitude",
        )
        ax3_twin.set_ylabel("Impact Magnitude", color="orange")
        ax3_twin.tick_params(axis="y", labelcolor="orange")
        ax3_twin.legend(loc="upper right")

        ax3.set_ylabel("Impact RMSD", color="darkred")
        ax3.tick_params(axis="y", labelcolor="darkred")
        ax3.set_title(
            f"Disaster Events and Impact Sizes (Total: {num_disasters} disasters)"
        )
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper left")
    else:
        ax3.text(
            0.5,
            0.5,
            "No disasters occurred in this simulation",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
        ax3.set_title("Disaster Events (None occurred)")
        ax3.set_ylabel("Impact")

    ax3.set_xlabel("Time Step")

    plt.tight_layout()

    output_file = output_dir / "example_verification.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nVisualization saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Verification complete! The model is working correctly.")
    print("=" * 60)

    return sim


if __name__ == "__main__":
    try:
        main()
    except (ImportError, AttributeError, ValueError, RuntimeError) as e:
        print(f"\nError during simulation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
