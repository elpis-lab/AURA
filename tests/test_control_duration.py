from __future__ import annotations

import numpy as np
import pytest

from experiment.task_time_efficiency import (
    task_execution_seconds,
    task_execution_step_seconds,
)
from methods.plan import (
    ControlEdge,
    duration_seconds_to_steps,
    duration_steps_to_seconds,
    validate_duration_range,
)


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_duration_round_trip(steps: int) -> None:
    seconds = duration_steps_to_seconds(steps, 0.125)
    assert duration_seconds_to_steps(
        seconds, 0.125, min_steps=1, max_steps=5
    ) == steps


@pytest.mark.parametrize(
    ("duration", "minimum", "maximum"),
    [(0.31, 1, 5), (0.0, 1, 5), (0.1, 2, 5), (0.6, 1, 5)],
)
def test_rejects_malformed_or_out_of_range_duration(
    duration: float, minimum: int, maximum: int
) -> None:
    with pytest.raises(ValueError):
        duration_seconds_to_steps(
            duration, 0.1, min_steps=minimum, max_steps=maximum
        )


def test_duration_range_must_be_ordered_and_positive() -> None:
    assert validate_duration_range(1, 2) == (1, 2)
    with pytest.raises(ValueError):
        validate_duration_range(0, 2)
    with pytest.raises(ValueError):
        validate_duration_range(3, 2)


def test_figure7_pushing_primitive_charges_two_physical_seconds() -> None:
    assert task_execution_step_seconds("pushing_object", 1.0) == 2.0
    assert task_execution_seconds([1, 1, 1, 1, 1], "pushing", 1.0) == 10.0
    assert task_execution_seconds([5, 1], "pushing_object", 1.0) == 12.0
    assert task_execution_seconds([5, 1], "kinematic_car", 1.0) == 6.0


def test_control_edge_is_immutable_and_keeps_duration_pair() -> None:
    edge = ControlEdge(
        source_state=np.zeros(3),
        target_state=np.ones(3),
        control=np.array([0.4, -0.1]),
        duration_steps=2,
        duration_seconds=0.4,
        source_vertex=7,
        target_vertex=9,
    )
    assert edge.edge_id == "7->9"
    assert edge.duration_steps == 2
    assert edge.duration_seconds == pytest.approx(0.4)
    with pytest.raises(ValueError):
        edge.control[0] = 2.0
