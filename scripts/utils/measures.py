import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy import stats
from pybdm import BDM
from typing import Sequence


def safe_binary_conversion(solution):
    """
    Safely convert single solution to binary
    Returns binary string or None if conversion fails
    """
    try:
        padded = str(solution).zfill(5)  # Ensure 5 digits
        return "".join([format(int(d), "04b") for d in padded])
    except ValueError:
        return None


def convert_solutions_to_binary(items_array):
    """Convert array of solutions to binary with validation"""
    binary_strings = []
    for solution in items_array:
        binary = safe_binary_conversion(solution)
        if binary is None:
            return None
        binary_strings.append(binary)
    return np.array([int(x) for x in "".join(binary_strings)])


def calculate_hamming_distances(items_array, sample_size=30):
    """Calculate sampled Hamming distances between items"""
    if len(items_array) <= 1:
        return np.array([0])

    n_items = len(items_array)
    if n_items > sample_size:
        idx1 = np.random.choice(n_items, sample_size)
        idx2 = np.random.choice(n_items, sample_size)
        pairs = zip([items_array[i] for i in idx1], [items_array[i] for i in idx2])
    else:
        pairs = [
            (items_array[i], items_array[j])
            for i in range(n_items)
            for j in range(i + 1, n_items)
        ]

    distances = [sum(c1 != c2 for c1, c2 in zip(str(s1), str(s2))) for s1, s2 in pairs]
    return np.array(distances)


def calculate_entropy(items_array):
    """Calculate Shannon entropy of item frequencies"""
    unique, counts = np.unique(items_array, return_counts=True)
    freqs = counts / len(items_array)
    return -np.sum(freqs * np.log2(freqs))


def calculate_nbdm(items_array):
    """Calculate Normalized block decomposition method"""
    bdm = BDM(ndim=1)
    return bdm.bdm(items_array)


# We introduce the Gini Coefficient as a measure of inequality in the distribution of disaster impacts by group
def gini_coefficient(x: Sequence[float]) -> float:
    """
    Calculate the Gini coefficient for an array of values.

    The Gini coefficient measures inequality by comparing each value with every other value.
    It ranges from 0 (perfect equality) to 1 (perfect inequality).

    Args:
        x: Array of non-negative values

    Returns:
        Float between 0 and 1
    """
    x = np.array(x)
    if len(x) < 2:
        return 0.0

    # Sort values
    sorted_x = np.sort(x)
    n = len(x)

    # Calculate cumulative proportion of population and values
    cumsum = np.cumsum(sorted_x)

    # Calculate Gini coefficient using area method
    # Perfect equality line: y = x
    # Actual distribution: Lorenz curve
    # Gini = area between curves / total area
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


################ 5. Calculate Cumulative Statistics ############################
def calculate_cumulative_values(group):
    """Calculate cumulative statistics using vectorized operations"""
    # Payoff statistics - using cumsum() for accumulating total values
    for stat in [
        "mean",
        "geomean",
        "median",
        "max",
        "min",
    ]:  # Include max and min in regular cumsum
        col = f"payoffs_{stat}"
        if not group[col].isna().all():
            group[f"{col}_cumulative"] = group[col].cumsum()

    # Special handling for std
    if not group["payoffs_std"].isna().all():
        group["payoffs_std_cumulative"] = np.sqrt(
            (group["payoffs_std"] ** 2).cumsum() / (np.arange(len(group)) + 1)
        )

    # Item diversity metrics
    for stat in ["unique_count", "entropy", "nbdm", "hamming_mean"]:
        col = f"items_{stat}"
        if not group[col].isna().all():
            group[f"{col}_cumulative"] = group[col].cumsum()

    # special handling for hamming std
    if not group["items_hamming_std"].isna().all():
        group["items_hamming_std_cumulative"] = np.sqrt(
            (group["items_hamming_std"] ** 2).cumsum() / (np.arange(len(group)) + 1)
        )

    # Disaster impact
    if not group["disaster_impact"].isna().all():
        group["disaster_impact_mean"] = np.mean(group["disaster_impact"])
        group["disaster_impact_std"] = np.std(group["disaster_impact"])
        group["disaster_impact_cumulative"] = group["disaster_impact"].cumsum()

    return group


def calculate_recovery_metrics(
    payoff_series, impact_series, window=5, threshold=0.01, max_recovery_time=50
):
    """
    Calculate metrics related to recovery after disruptions

    Args:
        payoff_series: Array of payoffs over time
        impact_series: Array of impact values (RMSD) over time
        window: Number of time steps to look ahead for recovery
        threshold: Minimum impact to consider as a disruption
        max_recovery_time: Maximum number of time steps to look ahead for full recovery

    Returns:
        Dictionary of recovery metrics
    """
    # Find disruption points
    disruption_indices = np.where(impact_series > threshold)[0]

    if len(disruption_indices) == 0:
        return {
            "count": 0,
            "avg_drop_pct": 0,
            "avg_recovery_pct": 0,
            "full_recovery_rate": 0,
            "avg_time_to_recovery": 0,
            "window": window,
            "threshold": threshold,
            "max_recovery_time": max_recovery_time,
        }

    drops = []
    recoveries = []
    full_recoveries = 0
    times_to_recovery = []
    interrupted_recoveries = 0  # Count recoveries interrupted by another disaster

    for idx in disruption_indices:
        # Skip if too close to beginning
        if idx <= 0:
            continue

        # Get pre and post disruption payoffs
        pre_value = payoff_series[idx - 1]
        post_value = payoff_series[idx]

        # Skip if no actual drop occurred
        if post_value >= pre_value:
            continue

        # Calculate percentage drop
        drop_pct = (pre_value - post_value) / pre_value * 100
        drops.append(drop_pct)

        # Check for additional disasters within recovery window
        recovery_target = pre_value
        recovery_window_end = min(idx + max_recovery_time, len(payoff_series))

        # Look for additional disasters in the recovery window
        additional_disasters = disruption_indices[
            (disruption_indices > idx) & (disruption_indices < recovery_window_end)
        ]

        # If there are additional disasters, use the highest pre-disaster value as recovery target
        if len(additional_disasters) > 0:
            for add_idx in additional_disasters:
                if add_idx > 0:  # Ensure we can look at pre-disaster value
                    add_pre_value = payoff_series[add_idx - 1]
                    recovery_target = max(recovery_target, add_pre_value)

        # Check recovery after window time steps
        if idx + window < len(payoff_series):
            recovery_value = payoff_series[idx + window]
            # compute raw recovery percent and clamp to [0,100]
            recovery_pct = (
                (recovery_value - post_value) / (recovery_target - post_value)
            ) * 100
            recovery_pct = max(min(recovery_pct, 100), 0)
            recoveries.append(recovery_pct)

            # Check if full recovery occurred within the window
            if recovery_value >= recovery_target:
                full_recoveries += 1
                times_to_recovery.append(window)  # Recovery happened within the window
            else:
                # Look ahead to find when full recovery occurs
                time_to_recovery = None
                recovery_interrupted = False

                for t in range(window + 1, recovery_window_end):
                    if t < len(payoff_series):
                        # Check if we hit another disaster before recovery
                        if t in additional_disasters:
                            recovery_interrupted = True
                            interrupted_recoveries += 1
                            break
                        if payoff_series[t] >= recovery_target:
                            time_to_recovery = t - idx
                            break

                if time_to_recovery is not None:
                    times_to_recovery.append(time_to_recovery)
                    full_recoveries += 1
                elif not recovery_interrupted:
                    # If we reached the end of the window without recovery and weren't interrupted,
                    # count it as not recovered
                    times_to_recovery.append(max_recovery_time)

    # Calculate summary statistics
    if len(drops) > 0:
        avg_time_to_recovery = (
            np.mean(times_to_recovery) if times_to_recovery else max_recovery_time
        )

        return {
            "count": len(drops),
            "avg_drop_pct": np.mean(drops),
            "avg_recovery_pct": np.mean(recoveries) if recoveries else 0,
            "full_recovery_rate": full_recoveries / len(drops),
            "interrupted_recovery_rate": interrupted_recoveries / len(drops),
            "avg_time_to_recovery": avg_time_to_recovery,
            "window": window,
            "threshold": threshold,
            "max_recovery_time": max_recovery_time,
        }
    else:
        return {
            "count": 0,
            "avg_drop_pct": 0,
            "avg_recovery_pct": 0,
            "full_recovery_rate": 0,
            "interrupted_recovery_rate": 0,
            "avg_time_to_recovery": 0,
            "window": window,
            "threshold": threshold,
            "max_recovery_time": max_recovery_time,
        }


def gini_coefficient(array):
    """
    Calculate the Gini coefficient of a numpy array.
    """
    if np.mean(array) == 0:  # Check for division by zero
        return 0.0

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(array, array)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(array)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def calculate_entropy(array):
    """
    Calculate the entropy of a numpy array.
    """
    unique, counts = np.unique(array, return_counts=True)
    probabilities = counts / len(array)
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_hamming_distances(array):
    """
    Calculate pairwise Hamming distances between all elements in array.
    """
    n = len(array)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = np.sum(array[i] != array[j])
            distances[j, i] = distances[i, j]
    return distances
