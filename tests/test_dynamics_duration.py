from __future__ import annotations

import numpy as np
import pytest
import torch
from ompl import control as oc

from propagators import (
    DoubleIntegrator,
    DubinsAirplane,
    KinematicCar,
    PushingObject,
    double_integrator,
    kinematic_car,
    pushing_object,
    get_system,
)


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("double_integrator", DoubleIntegrator),
        ("dubins_airplane", DubinsAirplane),
        ("kinematic_car", KinematicCar),
        ("pushing_object", PushingObject),
    ],
)
def test_system_factory_uses_canonical_names(name: str, expected_type: type) -> None:
    assert isinstance(get_system(name), expected_type)


@pytest.mark.parametrize("alias", ["car", "simple_car", "push", "dubins"])
def test_system_factory_rejects_legacy_aliases(alias: str) -> None:
    with pytest.raises(ValueError, match="Unknown system"):
        get_system(alias)


@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("steering", [0.0, 0.22])
def test_car_numpy_torch_and_semigroup(steps: int, steering: float) -> None:
    start = np.array([0.3, -0.4, 0.7])
    control = np.array([0.6, steering])
    h = 0.17
    expected = kinematic_car.propagate_numpy(start, control, steps * h)
    repeated = start.copy()
    for _ in range(steps):
        repeated = kinematic_car.propagate_numpy(repeated, control, h)
    actual = (
        kinematic_car.propagate_torch(
            torch.tensor(start[None], dtype=torch.float64),
            torch.tensor(control[None], dtype=torch.float64),
            torch.tensor([steps * h], dtype=torch.float64),
        )
        .detach()
        .numpy()[0]
    )
    np.testing.assert_allclose(expected, repeated, atol=1e-10)
    np.testing.assert_allclose(expected, actual, atol=1e-10)


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_double_integrator_numpy_torch_and_semigroup(steps: int) -> None:
    start = np.array([0.3, -0.4, 0.7, 0.1, -0.2, 0.05])
    control = np.array([0.08, -0.06, 0.1])
    h = 0.2
    expected = double_integrator.propagate_numpy(start, control, steps * h)
    repeated = start.copy()
    for _ in range(steps):
        repeated = double_integrator.propagate_numpy(repeated, control, h)
    actual = (
        double_integrator.propagate_torch(
            torch.tensor(start[None], dtype=torch.float64),
            torch.tensor(control[None], dtype=torch.float64),
            torch.tensor([steps * h], dtype=torch.float64),
        )
        .detach()
        .numpy()[0]
    )
    np.testing.assert_allclose(expected, repeated, atol=1e-12)
    np.testing.assert_allclose(expected, actual, atol=1e-12)


class _RelativePush(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._device_anchor = torch.nn.Parameter(
            torch.zeros(1), requires_grad=False
        )

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                0.4 * control[:, 2],
                0.1 * control[:, 1],
                0.2 * control[:, 1],
            ],
            dim=1,
        )


def _ompl_propagate(system, start: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
    setup = oc.SimpleSetup(system.control_space)
    si = setup.getSpaceInformation()
    si.setPropagationStepSize(float(system.propagation_step_size))
    propagator_wrapper = getattr(oc, "StatePropagatorFn", None)
    si.setStatePropagator(
        propagator_wrapper(system.ompl_propagator)
        if propagator_wrapper is not None
        else system.ompl_propagator
    )
    start_state = system.state_space.allocState()
    result_state = system.state_space.allocState()
    control_state = system.control_space.allocControl()
    if system.name in ("kinematic_car", "pushing_object"):
        start_state.setX(float(start[0]))
        start_state.setY(float(start[1]))
        start_state.setYaw(float(start[2]))
    else:
        for index, value in enumerate(start):
            start_state[index] = float(value)
    for index, value in enumerate(control):
        control_state[index] = float(value)
    steps = int(round(float(duration) / float(system.propagation_step_size)))
    si.propagate(start_state, control_state, steps, result_state)
    if system.name in ("kinematic_car", "pushing_object"):
        return np.array(
            [result_state.getX(), result_state.getY(), result_state.getYaw()],
            dtype=float,
        )
    return np.array([result_state[index] for index in range(len(start))], dtype=float)


@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("system_name", ["kinematic_car", "double_integrator"])
def test_ompl_propagator_matches_canonical_endpoint(
    steps: int, system_name: str
) -> None:
    h = 0.17
    if system_name == "kinematic_car":
        system = KinematicCar()
        start = np.array([0.3, -0.4, 0.7])
        control = np.array([0.6, 0.22])
    else:
        system = DoubleIntegrator()
        start = np.array([0.3, -0.4, 0.7, 0.1, -0.2, 0.05])
        control = np.array([0.08, -0.06, 0.1])
    system.configure_duration_contract(h, 1, 5)
    expected = system.propagate(start, control, steps * h)
    actual = _ompl_propagate(system, start, control, steps * h)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_pushing_ompl_propagator_honors_duration(
    steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _RelativePush().to(dtype=torch.float32)
    monkeypatch.setattr(
        "propagators.pushing_object.get_pushing_model",
        lambda *args, **kwargs: model,
    )
    system = PushingObject()
    system.configure_duration_contract(0.2, 1, 5)
    start = np.array([0.1, -0.2, 0.35])
    control = np.array([0.25, 0.18, 0.12])
    expected = system.propagate(start, control, steps * 0.2)
    actual = _ompl_propagate(system, start, control, steps * 0.2)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_pushing_unroll_composes_every_primitive(steps: int) -> None:
    start = np.array([0.1, -0.2, 0.35])
    control = np.array([0.25, 0.18, 0.12])
    model = _RelativePush()

    def one_step(batch: np.ndarray) -> np.ndarray:
        return model(torch.tensor(batch, dtype=torch.float64)).detach().numpy()

    expected = start.copy()
    delta = one_step(control[None])[0]
    for _ in range(steps):
        expected = pushing_object.compose_numpy(expected, delta)
    numpy_result = pushing_object.propagate_numpy(start, control, steps, one_step)
    torch_result = (
        pushing_object.propagate_torch(
            torch.tensor(start[None], dtype=torch.float64),
            torch.tensor(control[None], dtype=torch.float64),
            torch.tensor([steps]),
            model,
        )
        .detach()
        .numpy()[0]
    )
    np.testing.assert_allclose(numpy_result, expected, atol=1e-12)
    np.testing.assert_allclose(torch_result, expected, atol=1e-12)


def test_pushing_numpy_rollout_vectorizes_pose_batches() -> None:
    starts = np.array([[0.1, -0.2, 0.35], [0.2, -0.3, -0.15]])
    control = np.array([0.25, 0.18, 0.12])
    model = _RelativePush()

    def one_step(batch: np.ndarray) -> np.ndarray:
        return model(torch.tensor(batch, dtype=torch.float64)).detach().numpy()

    batched = pushing_object.propagate_numpy(starts, control, 3, one_step)
    separate = np.stack(
        [
            pushing_object.propagate_numpy(start, control, 3, one_step)
            for start in starts
        ]
    )
    np.testing.assert_allclose(batched, separate, atol=1e-12)


@pytest.mark.parametrize("system", ["car", "double_integrator", "pushing"])
def test_multistep_gradients_match_finite_difference(system: str) -> None:
    epsilon = 1e-6
    if system == "car":
        start = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
        control = torch.tensor([[0.55, 0.17]], dtype=torch.float64, requires_grad=True)
        fn = lambda u: kinematic_car.propagate_torch(start, u, 0.8).sum()
        continuous_indices = [0, 1]
    elif system == "double_integrator":
        start = torch.tensor(
            [[0.1, -0.2, 0.3, 0.04, -0.05, 0.06]], dtype=torch.float64
        )
        control = torch.tensor(
            [[0.07, -0.08, 0.09]], dtype=torch.float64, requires_grad=True
        )
        fn = lambda u: double_integrator.propagate_torch(start, u, 0.8).sum()
        continuous_indices = [0, 1, 2]
    else:
        model = _RelativePush()
        start = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
        control = torch.tensor(
            [[0.25, 0.16, 0.11]], dtype=torch.float64, requires_grad=True
        )
        fn = lambda u: pushing_object.propagate_torch(
            start, u, torch.tensor([4]), model
        ).sum()
        continuous_indices = [1, 2]

    analytic = torch.autograd.grad(fn(control), control)[0].detach().numpy()[0]
    assert np.all(np.isfinite(analytic))
    for index in continuous_indices:
        plus = control.detach().clone()
        minus = control.detach().clone()
        plus[0, index] += epsilon
        minus[0, index] -= epsilon
        numerical = float((fn(plus) - fn(minus)) / (2.0 * epsilon))
        assert analytic[index] == pytest.approx(numerical, rel=2e-5, abs=2e-6)
