#!/usr/bin/env python3
"""
Main simulation runner for the model.

This is the primary entry point for running simulations. It supports
single simulations, multiple repetitions, and parameter sweeps.
"""

import os
import sys
import logging
import itertools
import argparse
from dask.distributed import Client, LocalCluster
from pathlib import Path
from typing import Dict, List, Any

# Add the project root directory to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.core.simulation import Simulation
from scripts.utils.storage import ResultsStorage
import scripts.config.params as params


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_single_simulation(
    parameters: Dict[str, Any], run_id: int, storage: ResultsStorage
) -> None:
    """Run a single simulation and save results"""
    try:
        #  logging.info(
        #      f"Starting simulation with parameters: {parameters}, Run ID: {run_id}"
        #  )

        # Create and run simulation
        sim = Simulation(parameters)
        results = sim.runEfficientSL()

        # Save results
        storage.save_run(parameters, run_id, results)

        # logging.info(
        #     f"Completed simulation with parameters: {parameters}, Run ID: {run_id}"
        # )

    except Exception as e:
        logging.error(
            f"Error in simulation with parameters {parameters}, Run ID {run_id}: {e}"
        )
        raise


def run_simulations(
    parameter_combinations: List[Dict[str, Any]],
    num_repetitions: int,
    output_dir: str,
    num_workers: int = None,
) -> None:
    """Run multiple simulations in parallel using Dask"""
    # Setup storage
    storage = ResultsStorage(output_dir)

    # Setup Dask cluster
    if num_workers is None:
        num_workers = os.cpu_count()

    cluster = LocalCluster(
        n_workers=num_workers, threads_per_worker=1, memory_limit="4GB"
    )
    client = Client(cluster)

    total_simulations = len(parameter_combinations) * num_repetitions
    completed_simulations = 0

    try:
        # Create list of all tasks
        tasks = []
        for params in parameter_combinations:
            for run_id in range(num_repetitions):
                tasks.append((params, run_id))

        # Submit all tasks
        futures = []
        for params, run_id in tasks:
            future = client.submit(run_single_simulation, params, run_id, storage)
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            client.gather(future)
            completed_simulations += 1
            logging.info(
                f"Completed {completed_simulations} of {total_simulations} simulations."
            )

    finally:
        client.close()
        cluster.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run simulations")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions for each parameter combination",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of Dask workers (default: number of CPU cores)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/simulation_results",
        help="Directory to save simulation results",
    )
    parser.add_argument(
        "--use-parameters",
        action="store_true",
        help="Use parameters() function instead of get_parameters()",
    )
    args = parser.parse_args()

    setup_logging()

    # Get parameters based on command line argument
    if args.use_parameters:
        # Use parameters() for multiple combinations
        p = params.parameters()

        # Convert single values to lists
        param_grid = {
            k: v if isinstance(v, (list, tuple)) else [v] for k, v in p.items()
        }

        # Generate parameter combinations
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        # Convert to dictionaries
        param_combinations = [
            dict(zip(param_names, comb)) for comb in param_combinations
        ]

        logging.info(f"Generated {len(param_combinations)} parameter combinations")
    else:
        # Use get_parameters() for a single parameter set
        """
        override_params = {
            "numberOfAgents": 100,
            "numberOfAgentGroups": 2,
            "numberOfTimeSteps": 400,
            "N_NKmodel": 15,
            "K_NKmodel": 7,
            "p_erNetwork": 0.04,
            "disasterProbability": 0.01,
            "95th_percentile": 0.5,
        }"""
        p = params.get_parameters()
        param_combinations = [p]
        logging.info("Using single parameter set")

    logging.info(f"Running {args.repetitions} repetitions of each combination")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulations
    run_simulations(
        parameter_combinations=param_combinations,
        num_repetitions=args.repetitions,
        output_dir=str(output_dir),
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
