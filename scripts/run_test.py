#!/usr/bin/env python3
"""
Test simulation script with visualization.

This script runs a test simulation using parameters from params_test.py
and generates visualization plots. Can be run from the scripts directory
or from the root directory.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path if running from scripts directory
if Path(__file__).parent.name == "scripts":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import matplotlib.pyplot as plt
import numpy as np
import time
from scripts.core.simulation import Simulation
import scripts.params_test as params_test


# time the simulation
start = time.time()
p = params_test.parameters()

sim = Simulation(p)
sim.runEfficientSL()
end = time.time()

print(f"Time taken: {end - start} seconds")

# Create the results directory if it doesn't exist
results_dir = Path(__file__).parent.parent / "results" / "fig"
results_dir.mkdir(parents=True, exist_ok=True)

name = f"n{p['N_NKmodel']}_k{p['K_NKmodel']}_ag{p['numberOfAgentGroups']}_er{p['p_erNetwork']}_dis{p['disasterProbability']}_clust{p['disasterClusteredness']}_5th{p['5th_percentile']}_95th{p['95th_percentile']}"
outputdir = results_dir / f"{name}plot.png"
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Top plot
for agentIdx in range(sim.numberOfAgents):
    ax1.plot(sim.payoffsAgentsOverTime[agentIdx, :], linewidth=0.2, color="k")

ax1.plot(
    np.mean(sim.payoffsAgentsOverTime, axis=0),
    color="red",
    linewidth=2.5,
    label="Collective Performance",
)
ax1.set_ylabel("Performance")
ax1.set_ylim([0, 1.1])  # set the y-axis range to be between 0 and 1

ax2 = ax1.twinx()
ax2.plot(sim.uniqueItemsOverTime, color="green", linewidth=2.5, label="Unique Items")
ax2.set_ylabel("Unique Items")
ax2.set_ylim(
    0, max(sim.uniqueItemsOverTime) * 1.1
)  # set the y-axis range to be between 0 and the number of items

# plot the disaster impacts as vertical lines
for i in range(sim.ImpactRmsd.shape[0]):
    if np.any(sim.ImpactRmsd[i] > 0):
        ax1.axvline(x=i, color="blue", linewidth=1, linestyle="--")
        ax3.axvline(x=i, color="blue", linewidth=1, linestyle="--")
ax1.set_title(f"Performance over Time - params: {name}")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Bottom plot
ax3.plot(
    sim.learnedSolutions, color="blue", linewidth=2.5, label="Social Learning Events"
)
ax3.plot(sim.innovatedSolutions, color="red", linewidth=2.5, label="Innovation Events")
ax3.set_xlabel("Time")
ax3.set_ylabel("Number of Events")
ax3.set_title(f"Social Learning and Innovation Events over Time - params: {name}")
ax3.legend(loc="upper left")
ax3.grid(True)


# Adjust layout and save image to file in the results folder
plt.tight_layout()
plt.savefig(str(outputdir))
print(f"Plot saved to: {outputdir}")
# plt.show()  # Commented out for headless execution
