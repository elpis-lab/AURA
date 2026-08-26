"""Duration-aware batched recovery-control optimization."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import numpy as np
import torch

from propagators import (
    System,
    double_integrator,
    dubins_airplane,
    kinematic_car,
    pushing_object,
)
from propagators.propagator import wrap_angle_torch
from simulation.pushing_model import get_pushing_model
from simulation.pushing_model import CRACKER_BOX_FLIPPED_SHAPE
from utils.control_duration import (
    ControlEdge,
    duration_seconds_to_steps,
)
from utils.utils import log


def weighted_state_loss(
    system: str,
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return squared canonical state distance for every recovery row."""
    key = str(system).lower()
    residual = predicted - target
    if key in ("kinematic_car", "pushing_object"):
        residual = residual.clone()
        residual[:, 2] = wrap_angle_torch(residual[:, 2])
        translation = torch.linalg.vector_norm(residual[:, :2], dim=1)
        distance = translation + 0.5 * torch.abs(residual[:, 2])
        row_loss = distance.square()
    elif key == "double_integrator":
        row_loss = torch.sum(residual[:, :6].square(), dim=1)
    elif key == "dubins_airplane":
        residual = residual.clone()
        residual[:, 3] = wrap_angle_torch(residual[:, 3])
        row_loss = torch.sum(residual[:, :6].square(), dim=1)
    else:
        row_loss = torch.sum(residual.square(), dim=1)
    if reduction == "none":
        return row_loss
    if reduction == "sum":
        return row_loss.sum()
    return row_loss.mean()


def optimizer_device_info(requested_device: str | None = None) -> dict:
    """Return a tested Torch device and reproducibility metadata."""
    info = {
        "device": "cpu",
        "requested_device": requested_device or "auto",
        "cuda_available": False,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "device_count": 0,
        "device_name": "",
        "error": "",
    }
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if requested_device:
            device = torch.device(requested_device)
        elif info["cuda_available"]:
            device = torch.device(f"cuda:{int(torch.cuda.current_device())}")
        else:
            device = torch.device("cpu")
        torch.zeros(1, device=device)
        info["device"] = str(device)
        if device.type == "cuda":
            info["device_name"] = torch.cuda.get_device_name(device)
    except Exception as exc:
        info["device"] = "cpu"
        info["error"] = repr(exc)
    return info


def warmup_optimizer_device(
    system: str = "kinematic_car",
    requested_device: str | None = None,
) -> None:
    """Initialize kernels used by the timed optimizer."""
    info = optimizer_device_info(requested_device)
    device = torch.device(str(info["device"]))
    if device.type != "cuda":
        return
    dimensions = {
        "kinematic_car": (3, 2),
        "double_integrator": (6, 3),
        "dubins_airplane": (6, 3),
    }
    system = str(system).lower()
    state_dim, control_dim = dimensions.get(system, (3, 2))
    start = torch.zeros((256, state_dim), dtype=torch.float32, device=device)
    control = torch.zeros(
        (256, control_dim), dtype=torch.float32, device=device, requires_grad=True
    )
    duration = torch.ones(256, dtype=torch.float32, device=device)
    target = torch.zeros_like(start)
    optimizer = torch.optim.Adam([control], lr=0.01)
    dynamics: Callable = {
        "kinematic_car": kinematic_car.propagate_torch,
        "double_integrator": double_integrator.propagate_torch,
        "dubins_airplane": dubins_airplane.propagate_torch,
    }.get(system, kinematic_car.propagate_torch)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_state_loss(system, dynamics(start, control, duration), target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)


def validate_child_edges(
    child_edges: list[ControlEdge], integration_step_size: float
) -> list[ControlEdge]:
    """Validate the typed branch edges consumed by the optimizer."""
    step_size = float(integration_step_size)
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("integration_step_size must be finite and positive")
    edges = list(child_edges)
    for edge in edges:
        if not isinstance(edge, ControlEdge):
            raise TypeError(f"Expected ControlEdge, got {type(edge)!r}")
        observed_steps = duration_seconds_to_steps(edge.duration_seconds, step_size)
        if observed_steps != edge.duration_steps:
            raise ValueError(
                f"edge {edge.edge_id!r} has inconsistent duration: "
                f"steps={edge.duration_steps}, seconds={edge.duration_seconds}, "
                f"integration_step_size={step_size}"
            )
    return edges


def clamp_controls(
    controls: torch.Tensor,
    bounds: list[tuple[float, float]],
    *,
    original_controls: torch.Tensor,
    system: str,
) -> None:
    with torch.no_grad():
        for index, (low, high) in enumerate(bounds):
            controls[:, index].clamp_(float(low), float(high))
        if system == "pushing_object":
            # Contact face is discrete and belongs to the branch edge.
            controls[:, 0].copy_(original_controls[:, 0])
            controls[:, 1].copy_(
                torch.maximum(
                    torch.minimum(controls[:, 1], original_controls[:, 1] + 0.08),
                    original_controls[:, 1] - 0.08,
                )
            )
            controls[:, 2].copy_(
                torch.maximum(
                    torch.minimum(controls[:, 2], original_controls[:, 2] + 0.010),
                    original_controls[:, 2] - 0.010,
                )
            )
            controls[:, 2].clamp_(
                float(bounds[2][0]), float(bounds[2][1])
            )


def bounds_activity(
    controls: torch.Tensor,
    bounds: list[tuple[float, float]],
    atol: float = 1e-6,
) -> list[bool]:
    values = controls.detach().cpu().numpy()
    bound_array = np.asarray(bounds, dtype=values.dtype)
    at_lower_bound = np.isclose(
        values, bound_array[:, 0], rtol=0.0, atol=float(atol)
    )
    at_upper_bound = np.isclose(
        values, bound_array[:, 1], rtol=0.0, atol=float(atol)
    )
    return np.any(at_lower_bound | at_upper_bound, axis=1).tolist()


def optimize_controls(
    system: System,
    next_state,
    child_edges: list[ControlEdge],
    *,
    integration_step_size: float,
    model=None,
    num_states: int = 1000,
    position_std: float = 0.003,
    rotation_std: float = 0.05,
    velocity_std: float | None = None,
    num_steps: int = 25,
    learning_rate: float = 0.05,
    max_wall_time: float | None = None,
    stop_event=None,
    partial_result_callback=None,
    requested_device: str | None = None,
):
    """Optimize a constant recovery control for each sampled-state/child edge.

    Rows are child-major: all sampled starts for child 0, then child 1, etc.
    Every row carries the duration and edge identity of its child.
    """
    total_started = time.perf_counter()
    total_started_monotonic = time.monotonic()
    try:
        system_key = system.name
        edges = validate_child_edges(child_edges, integration_step_size)
        if not edges:
            return None

        device_info = optimizer_device_info(requested_device)
        if requested_device and str(device_info["device"]) != str(
            torch.device(requested_device)
        ):
            raise RuntimeError(
                f"Requested optimizer device {requested_device!r} is unavailable: "
                f"{device_info.get('error', '')}"
            )
        device = torch.device(str(device_info["device"]))
        using_cuda = device.type == "cuda"
        child_count = len(edges)
        requested_num_states = int(num_states)
        requested_num_steps = int(num_steps)
        if requested_num_states < 1 or requested_num_steps < 1:
            raise ValueError("num_states and num_steps must be positive")

        wall_budget = (
            None if max_wall_time is None else max(0.0, float(max_wall_time))
        )
        # max_wall_time is an end-to-end budget, including candidate
        # construction, initial/final prediction, and result packaging. Reserve
        # a small tail for result packaging.  The audited per-row loss for the
        # best iterate is cached below, so finalization does not repeat an
        # expensive dynamics rollout after the optimization deadline.
        deadline = None
        if wall_budget is not None:
            finalization_guard = min(0.15, max(0.001, 0.15 * wall_budget))
            deadline = total_started_monotonic + max(
                0.0, wall_budget - finalization_guard
            )
        max_duration_steps = max(edge.duration_steps for edge in edges)
        requested_rows = requested_num_states * child_count
        if wall_budget is None:
            max_rows = 60000
            max_steps_by_wall = requested_num_steps
        elif using_cuda:
            max_rows = int(np.clip(120000 * max(wall_budget, 0.25), 30000, 500000))
            # One second must include the initial nonlinear rollout, gradients,
            # and audited result packaging.  Empirical Stage-1 timing showed
            # that multiplying b=5000 by every tree child can consume the whole
            # window before the first Adam step for the new RK4 systems.  Keep
            # b=5000 for a single child and share a measured total-row budget
            # across children during online branch optimization.
            online_row_rates = {"dubins_airplane": 2500}
            if system_key in online_row_rates:
                max_rows = min(
                    max_rows,
                    max(
                        child_count,
                        int(online_row_rates[system_key] * max(wall_budget, 0.25)),
                    ),
                )
            max_steps_by_wall = requested_num_steps
        else:
            work_scale = max_duration_steps if system_key == "pushing_object" else 1
            max_rows = int(
                np.clip(
                    8000 * max(wall_budget, 0.25) / work_scale,
                    500,
                    40000,
                )
            )
            max_steps_by_wall = int(
                np.clip(25 * max(wall_budget, 0.25), 6, 120)
            )
        effective_rows = max(child_count, min(requested_rows, max_rows))
        effective_num_states = max(1, effective_rows // child_count)
        effective_num_steps = max(1, min(requested_num_steps, max_steps_by_wall))

        candidate_started = time.perf_counter()
        sampled_states = system.sample_random_states(
            next_state,
            effective_num_states,
            position_std,
            rotation_std,
            velocity_std,
        )
        sampled_states = np.asarray(sampled_states, dtype=float)
        measured_state = np.asarray(next_state, dtype=float).reshape(-1)
        if sampled_states.ndim != 2 or sampled_states.shape[0] != effective_num_states:
            raise ValueError("system state sampler returned an invalid batch shape")
        if sampled_states.shape[1] > measured_state.size:
            raise ValueError("measured state is smaller than sampled state dimension")
        # Always optimize one control from the exact measured state. Random
        # neighborhood coverage must not determine whether recovery is possible
        # at the state that will actually execute the selected control.
        sampled_states[0] = measured_state[: sampled_states.shape[1]]
        targets = np.asarray([edge.target_state for edge in edges], dtype=float)
        original_child_controls = np.asarray(
            [edge.control for edge in edges], dtype=float
        )
        child_seconds = np.asarray(
            [edge.duration_seconds for edge in edges], dtype=float
        )
        child_steps = np.asarray([edge.duration_steps for edge in edges], dtype=int)

        repeated_seconds = np.repeat(
            child_seconds, effective_num_states, axis=0
        )
        repeated_steps = np.repeat(child_steps, effective_num_states, axis=0)
        repeated_edge_ids = [
            edge.edge_id
            for edge in edges
            for _ in range(effective_num_states)
        ]

        start_states = torch.as_tensor(
            sampled_states, dtype=torch.float32, device=device
        ).repeat((child_count, 1))
        target_states = torch.as_tensor(
            targets, dtype=torch.float32, device=device
        ).repeat_interleave(effective_num_states, dim=0)
        original_controls = torch.as_tensor(
            original_child_controls, dtype=torch.float32, device=device
        ).repeat_interleave(effective_num_states, dim=0)
        optimized_controls = original_controls.detach().clone().requires_grad_(True)
        duration_seconds = torch.as_tensor(
            child_seconds, dtype=torch.float32, device=device
        ).repeat_interleave(effective_num_states)
        duration_steps = torch.as_tensor(
            child_steps, dtype=torch.long, device=device
        ).repeat_interleave(effective_num_states)
        bounds = list(system.control_bounds)
        if len(bounds) != original_child_controls.shape[1]:
            raise ValueError(
                f"{system_key} optimizer received {len(bounds)} bounds for "
                f"{original_child_controls.shape[1]} control dimensions"
            )

        pushing_model = None
        if system_key == "pushing_object":
            pushing_model = getattr(model, "model", None)
            if pushing_model is None:
                pushing_model = get_pushing_model(CRACKER_BOX_FLIPPED_SHAPE)
            pushing_model = pushing_model.to(device)
            pushing_model.eval()
            for parameter in pushing_model.parameters():
                parameter.requires_grad_(False)

        if using_cuda:
            torch.cuda.synchronize(device)
        candidate_batch_seconds = time.perf_counter() - candidate_started

        if system_key == "pushing_object":

            def predict(controls: torch.Tensor) -> torch.Tensor:
                return pushing_object.propagate_torch(
                    start_states, controls, duration_steps, pushing_model
                )

        else:
            propagate_torch = {
                "kinematic_car": kinematic_car.propagate_torch,
                "double_integrator": double_integrator.propagate_torch,
                "dubins_airplane": dubins_airplane.propagate_torch,
            }.get(system_key)
            if propagate_torch is None:
                raise ValueError(f"Unsupported optimizer system: {system_key}")

            def predict(controls: torch.Tensor) -> torch.Tensor:
                return propagate_torch(start_states, controls, duration_seconds)

        optimization_started = time.perf_counter()
        with torch.no_grad():
            original_row_loss_tensor = weighted_state_loss(
                system_key,
                predict(original_controls),
                target_states,
                reduction="none",
            )
        optimizer = torch.optim.Adam([optimized_controls], lr=float(learning_rate))
        loss_history: list[float] = []
        best_loss = float(original_row_loss_tensor.mean().detach().cpu())
        best_controls = original_controls.detach().clone()
        best_row_loss_tensor = original_row_loss_tensor.detach().clone()
        timed_out = False

        def make_result(*, partial: bool) -> dict:
            if using_cuda:
                torch.cuda.synchronize(device)
            final_row_loss = best_row_loss_tensor
            optimized_cpu = best_controls.detach().cpu()
            optimized_numpy = optimized_cpu.numpy()
            original_loss = original_row_loss_tensor.detach().cpu().numpy()
            optimized_loss = final_row_loss.detach().cpu().numpy()
            finite_rows = (
                np.isfinite(optimized_numpy).all(axis=1)
                & np.isfinite(optimized_loss)
            )
            return {
                "optimized_controls": optimized_cpu,
                "start_states": start_states.detach().cpu(),
                "target_states": target_states.detach().cpu(),
                "duration_seconds": duration_seconds.detach().cpu(),
                "duration_steps": duration_steps.detach().cpu(),
                "edge_ids": list(repeated_edge_ids),
                "child_edges": [edge.as_dict() for edge in edges],
                "row_metadata": [
                    {
                        "edge_id": repeated_edge_ids[index],
                        "duration_steps": int(repeated_steps[index]),
                        "duration_seconds": float(repeated_seconds[index]),
                        "original_loss": float(original_loss[index]),
                        "optimized_loss": float(optimized_loss[index]),
                        "finite": bool(finite_rows[index]),
                    }
                    for index in range(len(repeated_edge_ids))
                ],
                "bounds_active": bounds_activity(best_controls, bounds),
                "optimization_success": bool(np.all(finite_rows)),
                "loss_history": list(loss_history),
                "initial_loss": float(original_row_loss_tensor.mean().detach().cpu()),
                "final_loss": float(final_row_loss.mean().detach().cpu()),
                "timed_out": timed_out,
                "stopped": bool(stop_event is not None and stop_event.is_set()),
                "partial_result": bool(partial),
                "steps_completed": len(loss_history),
                "requested_num_states": requested_num_states,
                "effective_num_states": effective_num_states,
                "requested_num_steps": requested_num_steps,
                "effective_num_steps": effective_num_steps,
                "num_children": child_count,
                "max_duration_steps": max_duration_steps,
                "wall_time_budget": wall_budget,
                "device": str(device),
                "cuda_available": bool(device_info.get("cuda_available", False)),
                "device_name": str(device_info.get("device_name", "")),
                "candidate_batch_construction_seconds": float(
                    candidate_batch_seconds
                ),
                "dynamics_gradient_seconds": float(
                    time.perf_counter() - optimization_started
                ),
                "total_optimizer_seconds": float(
                    time.perf_counter() - total_started
                ),
                "candidate_batch_shape": {
                    "start_states": list(start_states.shape),
                    "target_states": list(target_states.shape),
                    "controls": list(optimized_controls.shape),
                    "durations": list(duration_seconds.shape),
                },
            }

        for step_index in range(effective_num_steps):
            if stop_event is not None and stop_event.is_set():
                timed_out = True
                break
            if deadline is not None and step_index > 0 and time.monotonic() >= deadline:
                timed_out = True
                break
            optimizer.zero_grad(set_to_none=True)
            predicted = predict(optimized_controls)
            loss = weighted_state_loss(system_key, predicted, target_states)
            if not torch.isfinite(loss):
                log("[WARNING] Recovery optimizer produced non-finite loss", "warning")
                break
            loss.backward()
            if system_key == "pushing_object" and optimized_controls.grad is not None:
                optimized_controls.grad[:, 0].zero_()
            optimizer.step()
            clamp_controls(
                optimized_controls,
                bounds,
                original_controls=original_controls,
                system=system_key,
            )
            with torch.no_grad():
                updated_row_loss = weighted_state_loss(
                    system_key,
                    predict(optimized_controls),
                    target_states,
                    reduction="none",
                )
                updated_loss = updated_row_loss.mean()
            loss_value = float(updated_loss.detach().cpu())
            loss_history.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_controls = optimized_controls.detach().clone()
                best_row_loss_tensor = updated_row_loss.detach().clone()

            if partial_result_callback is not None and (
                step_index == 0
                or step_index + 1 == effective_num_steps
                or (step_index + 1) % (8 if using_cuda else 1) == 0
            ):
                partial_result_callback(make_result(partial=True))
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break

        result = make_result(partial=False)
        if using_cuda:
            torch.cuda.synchronize(device)
        finished = time.perf_counter()
        result["dynamics_gradient_seconds"] = float(
            finished - optimization_started
        )
        result["total_optimizer_seconds"] = float(finished - total_started)
        return result
    except Exception as exc:
        log(f"[ERROR] in optimize_controls: {exc}", "error")
        import traceback

        traceback.print_exc()
        return None
