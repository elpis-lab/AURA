from __future__ import annotations

import numpy as np
import pytest

from simulation.simulator import (
    DoubleIntegratorGaussianNoise,
    KinematicCarGaussianNoise,
)


def test_gaussian_execution_applies_and_records_every_primitive() -> None:
    simulator = KinematicCarGaussianNoise(
        {
            "start_state": [0.0, 0.0, 0.0],
            "propagation_step_size": 0.2,
            "min_control_duration": 1,
            "max_control_duration": 4,
            "sampling_position_std": 0.0,
            "sampling_rotation_std": 0.0,
            "disturbance_seed": 9,
        }
    )
    simulator.execute_segment([0.5, 0.0], 0.6)
    assert len(simulator.primitive_states) == 4
    np.testing.assert_allclose(simulator.current_state, [0.3, 0.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        simulator.execute_segment([0.5, 0.0], 0.65)


def test_paired_methods_consume_identical_disturbance_prefix() -> None:
    config = {
        "start_state": [0.0] * 6,
        "propagation_step_size": 0.1,
        "min_control_duration": 1,
        "max_control_duration": 4,
        "sampling_velocity_std": 0.03,
        "disturbance_seed": 4711,
    }
    first = DoubleIntegratorGaussianNoise(config)
    second = DoubleIntegratorGaussianNoise(config)
    first.execute_segment([0.0, 0.0, 0.0], 0.3)
    second.execute_segment([0.0, 0.0, 0.0], 0.1)
    second.execute_segment([0.0, 0.0, 0.0], 0.2)
    np.testing.assert_allclose(first.primitive_states, second.primitive_states, atol=0.0)
    assert first._disturbance_index == second._disturbance_index == 3
