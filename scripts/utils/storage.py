import os
import xarray as xr
import numpy as np
from pathlib import Path
import logging


class ResultsStorage:
    def __init__(self, output_dir):
        """Initialize storage with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_path(self, params, run_id):
        """Get path for storing results of a specific run."""
        # Create a hash of parameters to use in filename
        param_str = str(sorted(params.items()))
        param_hash = str(hash(param_str))
        param_dir = self.output_dir / param_hash
        param_dir.mkdir(exist_ok=True)
        return param_dir / f"run_{run_id}.zarr"

    def save_run(self, params, run_id, results):
        """Save simulation results using xarray and Zarr."""
        output_path = self._get_run_path(params, run_id)

        # Check if the output path already exists
        if os.path.exists(output_path):
            logging.warning(
                f"Output path {output_path} already exists. Overwriting results."
            )
            # Just delete the existing directory and recreate it
            import shutil

            shutil.rmtree(output_path, ignore_errors=True)

        # Create time dimension
        time = np.arange(len(results["time_series.payoffs.mean"]))

        # Create datasets for different types of data
        time_series_ds = xr.Dataset(
            {
                "payoff_mean": (["time"], results["time_series.payoffs.mean"]),
                "payoff_std": (["time"], results["time_series.payoffs.std"]),
                "payoff_min": (["time"], results["time_series.payoffs.min"]),
                "payoff_max": (["time"], results["time_series.payoffs.max"]),
                "payoff_median": (["time"], results["time_series.payoffs.median"]),
                "unique_items": (["time"], results["time_series.items.unique_count"]),
                "item_entropy": (["time"], results["time_series.items.entropy"]),
                "item_hamming_mean": (
                    ["time"],
                    results["time_series.items.hamming_mean"],
                ),
                "social_learning": (
                    ["time"],
                    results["time_series.events.social_learning"],
                ),
                "innovation": (["time"], results["time_series.events.innovation"]),
                "impact_rmsd": (["time"], results["time_series.impact.rmsd"]),
                "impact_correlation": (
                    ["time"],
                    results["time_series.impact.correlation"],
                ),
                "impact_magnitude": (["time"], results["time_series.impact.magnitude"]),
            },
            coords={"time": time},
            attrs={"description": "Time series data from simulation"},
        )

        # Create summary dataset
        summary_ds = xr.Dataset(
            {
                "final_payoff_mean": ((), results["summary.final.payoff_mean"]),
                "final_payoff_median": ((), results["summary.final.payoff_median"]),
                "final_item_diversity": ((), results["summary.final.item_diversity"]),
                "cumulative_payoff_mean": (
                    (),
                    results["summary.cumulative.payoff_mean"],
                ),
                "cumulative_payoff_median": (
                    (),
                    results["summary.cumulative.payoff_median"],
                ),
                "cumulative_unique_items": (
                    (),
                    results["summary.cumulative.unique_items"],
                ),
                "cumulative_social_learning": (
                    (),
                    results["summary.cumulative.social_learning"],
                ),
                "cumulative_innovation": ((), results["summary.cumulative.innovation"]),
                "per_timestep_innovation": (
                    (),
                    results["summary.per_timestep.innovation"],
                ),
                "per_timestep_social_learning": (
                    (),
                    results["summary.per_timestep.social_learning"],
                ),
                "ratio_innovation": ((), results["summary.ratio.innovation"]),
                "average_payoff_mean": ((), results["summary.average.payoff_mean"]),
                "average_item_diversity": (
                    (),
                    results["summary.average.item_diversity"],
                ),
                "variability_payoff": ((), results["summary.variability.payoff"]),
                "variability_item_diversity": (
                    (),
                    results["summary.variability.item_diversity"],
                ),
            },
            attrs={"description": "Summary metrics from simulation"},
        )

        # Create disaster dataset
        disaster_ds = xr.Dataset(
            {
                "count": ((), results["disaster.count"]),
                "density": ((), results["disaster.density"]),
                "total_impact_rmsd": ((), results["disaster.total_impact_rmsd"]),
                "total_impact_magnitude": (
                    (),
                    results["disaster.total_impact_magnitude"],
                ),
                "average_impact_rmsd": ((), results["disaster.average_impact_rmsd"]),
                "average_impact_correlation": (
                    (),
                    results["disaster.average_impact_correlation"],
                ),
                "average_impact_magnitude": (
                    (),
                    results["disaster.average_impact_magnitude"],
                ),
                "max_impact_rmsd": ((), results["disaster.max_impact_rmsd"]),
                "max_impact_magnitude": ((), results["disaster.max_impact_magnitude"]),
                "average_gini_rmsd": (
                    (),
                    results["disaster.average_gini_rmsd"],
                ),
                "average_gini_correlation": (
                    (),
                    results["disaster.average_gini_correlation"],
                ),
                "average_gini_magnitude": (
                    (),
                    results["disaster.average_gini_magnitude"],
                ),
                "max_gini_rmsd": ((), results["disaster.max_gini_rmsd"]),
            },
            attrs={"description": "Disaster metrics from simulation"},
        )

        # Create recovery dataset
        recovery_ds = xr.Dataset(
            {
                "count": ((), results["recovery.metrics"]["count"]),
                "avg_drop_pct": ((), results["recovery.metrics"]["avg_drop_pct"]),
                "avg_recovery_pct": (
                    (),
                    results["recovery.metrics"]["avg_recovery_pct"],
                ),
                "full_recovery_rate": (
                    (),
                    results["recovery.metrics"]["full_recovery_rate"],
                ),
                "avg_time_to_recovery": (
                    (),
                    results["recovery.metrics"]["avg_time_to_recovery"],
                ),
                "window": ((), results["recovery.metrics"]["window"]),
                "threshold": ((), results["recovery.metrics"]["threshold"]),
                "max_recovery_time": (
                    (),
                    results["recovery.metrics"]["max_recovery_time"],
                ),
            },
            attrs={"description": "Recovery metrics from simulation"},
        )

        # Create network dataset
        network_ds = xr.Dataset(
            {
                "mean_degree": ((), results["network.mean_degree"]),
                "std_degree": ((), results["network.std_degree"]),
                "mean_clustering": ((), results["network.mean_clustering"]),
                "global_efficiency": ((), results["network.global_efficiency"]),
                "density": ((), results["network.density"]),
                "assortativity": ((), results["network.assortativity"]),
                "avg_path_length": ((), results["network.avg_path_length"]),
                "degree_percentile_25": ((), results["network.degree_percentile_25"]),
                "degree_percentile_50": ((), results["network.degree_percentile_50"]),
                "degree_percentile_75": ((), results["network.degree_percentile_75"]),
                # Group network properties
                "group_mean_degree": (("group",), results["network.group.mean_degree"]),
                "group_mean_betweenness": (
                    ("group",),
                    results["network.group.mean_betweenness"],
                ),
                "group_mean_clustering": (
                    ("group",),
                    results["network.group.mean_clustering"],
                ),
                "group_homophily": (("group",), results["network.group.homophily"]),
                "group_external_internal": (
                    ("group",),
                    results["network.group.external_internal"],
                ),
                "group_path_length_other_groups": (
                    ("group",),
                    results["network.group.path_length_other_groups"],
                ),
            },
            attrs={"description": "Network properties from simulation"},
        )

        # Create metadata dataset
        meta_ds = xr.Dataset(
            {
                "parameters": ((), str(results["meta.parameters"])),
                "N_NK": ((), results["meta.N_NK"]),
                "K_NK": ((), results["meta.K_NK"]),
                "p_er_network": ((), results["meta.p_er_network"]),
                "number_of_agents": ((), results["meta.number_of_agents"]),
                "number_of_groups": ((), results["meta.number_of_groups"]),
                "number_of_timesteps": ((), results["meta.number_of_timesteps"]),
            },
            attrs={"description": "Simulation metadata and parameters"},
        )

        # Save all datasets to Zarr
        time_series_ds.to_zarr(output_path / "time_series")
        summary_ds.to_zarr(output_path / "summary")
        disaster_ds.to_zarr(output_path / "disaster")
        recovery_ds.to_zarr(output_path / "recovery")
        network_ds.to_zarr(output_path / "network")
        meta_ds.to_zarr(output_path / "meta")

    def load_run(self, params, run_id):
        """Load simulation results from Zarr storage."""
        base_path = self._get_run_path(params, run_id)

        # Load all datasets
        time_series = xr.open_zarr(base_path / "time_series")
        summary = xr.open_zarr(base_path / "summary")
        disaster = xr.open_zarr(base_path / "disaster")
        recovery = xr.open_zarr(base_path / "recovery")
        network = xr.open_zarr(base_path / "network")
        meta = xr.open_zarr(base_path / "meta")

        return {
            "time_series": time_series,
            "summary": summary,
            "disaster": disaster,
            "recovery": recovery,
            "network": network,
            "meta": meta,
        }

    def run_exists(self, params, run_id):
        """
        Check if a run exists for a given parameter set and run ID.

        Parameters:
        -----------
        params : Dict[str, Any]
            Parameter set
        run_id : int
            Run ID

        Returns:
        --------
        bool
            True if the run exists, False otherwise
        """
        base_path = self._get_run_path(params, run_id)
        return base_path.exists() and (base_path / "time_series").exists()
