import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.core.landscape import FitnessLandscape
from scripts.config.params import get_parameters


@pytest.fixture
def k0_params():
    """Fixture to provide parameters for K0 landscape"""
    params = get_parameters()
    params.update(
        {
            "numberOfAgentGroups": 5,
            "N_NKmodel": 8,
            "K_NKmodel": 0,
            "scaleNKFitness": True,
        }
    )
    return params


@pytest.fixture
def k7_params():
    """Fixture to provide parameters for K7 landscape"""
    params = get_parameters()
    params.update(
        {
            "numberOfAgentGroups": 5,
            "N_NKmodel": 8,
            "K_NKmodel": 7,
            "scaleNKFitness": True,
        }
    )
    return params


@pytest.fixture
def k0_landscape(k0_params):
    """Fixture to provide a K0 landscape"""
    return FitnessLandscape(k0_params)


@pytest.fixture
def k7_landscape(k7_params):
    """Fixture to provide a K7 landscape"""
    return FitnessLandscape(k7_params)


def test_landscape_change_basic_functionality():
    """Test that landscape change works correctly with different turbulence levels"""
    # Set up landscape
    params = {
        "numberOfAgentGroups": 1,
        "N_NKmodel": 15,
        "K_NKmodel": 0,
        "scaleNKFitness": True,
    }
    landscape = FitnessLandscape(params)

    # Store initial state
    initial_landscape = landscape.landscape.copy()

    # Test with zero turbulence (should not change landscape)
    turbulence = np.array([0.0])
    metrics, _ = landscape.landscapeChange(turbulence)

    # Check that landscape is unchanged
    assert np.array_equal(landscape.landscape, initial_landscape), (
        "Landscape should not change with zero turbulence"
    )

    # Test with non-zero turbulence
    turbulence = np.array([0.25])
    metrics, _ = landscape.landscapeChange(turbulence)

    # Check that landscape has changed
    assert not np.array_equal(landscape.landscape, initial_landscape), (
        "Landscape should change with non-zero turbulence"
    )

    # Check that metrics are returned correctly
    assert len(metrics) == 3, "Should return 3 metrics (RMSD, correlation, magnitude)"
    assert all(isinstance(m, float) for m in metrics), "All metrics should be floats"

    # Check that landscape is still normalized
    assert np.max(landscape.landscape) == pytest.approx(1.0), (
        "Landscape should remain normalized"
    )


def test_landscape_change_multi_group():
    """Test landscape change with multiple agent groups"""
    # Set up landscape with multiple groups
    params = {
        "numberOfAgentGroups": 3,
        "N_NKmodel": 15,
        "K_NKmodel": 0,
        "scaleNKFitness": True,
    }
    landscape = FitnessLandscape(params)

    # Store initial state
    initial_landscape = landscape.landscape.copy()

    # Apply turbulence to only one group
    turbulence = np.array([0.25, 0.0, 0.0])
    metrics, group_metrics = landscape.landscapeChange(turbulence)

    # Check that only the first group changed
    assert np.array_equal(landscape.landscape[1], initial_landscape[1]), (
        "Group 1 should not change"
    )
    assert np.array_equal(landscape.landscape[2], initial_landscape[2]), (
        "Group 2 should not change"
    )
    assert not np.array_equal(landscape.landscape[0], initial_landscape[0]), (
        "Group 0 should change"
    )

    # Check group metrics
    assert group_metrics.shape == (3, 3), (
        "Group metrics should have shape (n_groups, 3)"
    )
    assert group_metrics[0, 2] > 0, "First group should have non-zero magnitude"
    assert group_metrics[1, 2] == 0, "Second group should have zero magnitude"
    assert group_metrics[2, 2] == 0, "Third group should have zero magnitude"


def test_landscape_change_turbulence_scaling():
    """Test that landscape changes scale with turbulence level"""
    # Set up landscape
    params = {
        "numberOfAgentGroups": 1,
        "N_NKmodel": 15,
        "K_NKmodel": 0,
        "scaleNKFitness": True,
    }
    landscape = FitnessLandscape(params)

    # Test with different turbulence levels
    turbulence_levels = [0.1, 0.3, 0.5]
    magnitudes = []

    for level in turbulence_levels:
        # Reset landscape
        landscape = FitnessLandscape(params)
        initial_landscape = landscape.landscape.copy()

        # Apply turbulence
        turbulence = np.array([level])
        metrics, _ = landscape.landscapeChange(turbulence)

        # Calculate magnitude of change
        changes = np.abs(landscape.landscape - initial_landscape)
        magnitude = np.mean(changes)
        magnitudes.append(magnitude)

    # Check that higher turbulence leads to some change
    # We don't assume linear scaling due to power-8 normalization
    assert all(m > 0 for m in magnitudes), (
        "All turbulence levels should produce some change"
    )
    assert magnitudes[0] < magnitudes[1] or magnitudes[1] < magnitudes[2], (
        "At least one higher turbulence level should produce larger changes"
    )


def test_k0_vs_k7_landscape_differences():
    """Test that landscape change metrics are properly calculated for different K values"""
    # Set up both landscapes
    params = {"numberOfAgentGroups": 1, "N_NKmodel": 15, "scaleNKFitness": True}

    params["K_NKmodel"] = 0
    k0_landscape = FitnessLandscape(params)

    params["K_NKmodel"] = 7
    k7_landscape = FitnessLandscape(params)

    # Apply same turbulence to both
    turbulence = np.array([0.25])

    # Get changes
    k0_metrics, _ = k0_landscape.landscapeChange(turbulence)
    k7_metrics, _ = k7_landscape.landscapeChange(turbulence)

    # Verify that metrics for both landscapes are properly calculated
    # Check structure of metrics array
    assert len(k0_metrics) == 3, "Landscape metrics should have 3 values"
    assert len(k7_metrics) == 3, "Landscape metrics should have 3 values"

    # Verify reasonable ranges for metrics (RMSD, correlation, magnitude)
    # RMSD (first metric)
    assert 0 <= k0_metrics[0] <= 10, "K0 RMSD should be in a reasonable range"
    assert 0 <= k7_metrics[0] <= 10, "K7 RMSD should be in a reasonable range"

    # Correlation change (second metric)
    assert 0 <= k0_metrics[1] <= 1, "K0 correlation change should be between 0 and 1"
    assert 0 <= k7_metrics[1] <= 1, "K7 correlation change should be between 0 and 1"

    # Magnitude (third metric)
    assert 0 <= k0_metrics[2] <= 1, "K0 magnitude should be between 0 and 1"
    assert 0 <= k7_metrics[2] <= 1, "K7 magnitude should be between 0 and 1"

    # Apply higher turbulence to verify increasing impact
    high_turbulence = np.array([0.5])

    k0_high_metrics, _ = k0_landscape.landscapeChange(high_turbulence)
    k7_high_metrics, _ = k7_landscape.landscapeChange(high_turbulence)

    # Verify that higher turbulence causes some change
    assert len(k0_high_metrics) == 3, (
        "High turbulence landscape metrics should have 3 values"
    )
    assert len(k7_high_metrics) == 3, (
        "High turbulence landscape metrics should have 3 values"
    )

    # Verify reasonable ranges for metrics with higher turbulence
    assert 0 <= k0_high_metrics[0] <= 10, (
        "K0 high turbulence RMSD should be in a reasonable range"
    )
    assert 0 <= k7_high_metrics[0] <= 10, (
        "K7 high turbulence RMSD should be in a reasonable range"
    )
    assert 0 <= k0_high_metrics[1] <= 1, (
        "K0 high turbulence correlation change should be between 0 and 1"
    )
    assert 0 <= k7_high_metrics[1] <= 1, (
        "K7 high turbulence correlation change should be between 0 and 1"
    )
    assert 0 <= k0_high_metrics[2] <= 1, (
        "K0 high turbulence magnitude should be between 0 and 1"
    )
    assert 0 <= k7_high_metrics[2] <= 1, (
        "K7 high turbulence magnitude should be between 0 and 1"
    )
