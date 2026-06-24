from __future__ import annotations

import os
import time

import numpy as np
import torch

from geometry.pose import SE2Pose
from systems import get_system
from utils.childrenHandler import sampleRandomState
from utils.utils import log


def optimizer_device_info() -> dict:
    """Return the torch device the dynamic optimizer will use."""
    info = {
        "device": "cpu",
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
        if info["cuda_available"]:
            index = int(torch.cuda.current_device())
            torch.zeros(1, device=f"cuda:{index}")
            info["device"] = f"cuda:{index}"
            info["device_name"] = torch.cuda.get_device_name(index)
    except Exception as exc:
        info["device"] = "cpu"
        info["error"] = repr(exc)
    return info


def warmup_optimizer_device(system: str = "kinematic_car") -> None:
    """Initialize CUDA kernels used by the optimizer before AURA's timed loop."""
    info = optimizer_device_info()
    if not str(info.get("device", "cpu")).startswith("cuda"):
        return
    device = torch.device(str(info["device"]))
    start = torch.zeros((256, 3), dtype=torch.float32, device=device)
    control = torch.zeros((256, 2), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.zeros((256, 3), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([control], lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        predicted = _kinematic_car_dynamics_torch(
            start,
            control,
            duration=1.0,
            integration_step_size=0.05,
        )
        loss = torch.nn.functional.mse_loss(predicted, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)


def _sample_random_states(
    system,
    state,
    numStates=1000,
    posSTD=0.003,
    rotSTD=0.05,
    velSTD=None,
):
    system_key = {
        "kinematic_car": "kinematic_car",
        "pushing_object": "pushing_object",
        "double_integrator": "double_integrator",
        "simple_car": "kinematic_car",
        "pushing": "pushing_object",
    }.get(system, system)

    sampled_states = []
    state_list = np.asarray(state, dtype=float).reshape(-1).tolist()

    if system_key in ("kinematic_car", "pushing_object"):
        for _ in range(numStates):
            noisy_x = state_list[0] + np.random.normal(0.0, posSTD)
            noisy_y = state_list[1] + np.random.normal(0.0, posSTD)
            noisy_yaw = state_list[2] + np.random.normal(0.0, rotSTD)
            while noisy_yaw > np.pi:
                noisy_yaw -= 2 * np.pi
            while noisy_yaw < -np.pi:
                noisy_yaw += 2 * np.pi
            sampled_states.append([noisy_x, noisy_y, noisy_yaw])
        return sampled_states

    if system_key == "double_integrator":
        if len(state_list) < 6:
            raise ValueError(
                f"double_integrator expects state with at least 6 values, got {len(state_list)}"
            )
        velocity_std = float(posSTD if velSTD is None else velSTD)
        for _ in range(numStates):
            noisy = state_list.copy()
            for i in range(3):
                noisy[i] += np.random.normal(0.0, posSTD)
            for i in range(3, 6):
                noisy[i] += np.random.normal(0.0, velocity_std)
            sampled_states.append(noisy[:6])
        return sampled_states

    return sampleRandomState(system, state, numStates, posSTD, rotSTD)


def _kinematic_car_dynamics_torch(
    start: torch.Tensor,
    control: torch.Tensor,
    *,
    duration: float,
    integration_step_size: float | None = None,
) -> torch.Tensor:
    wheelbase = 0.1385 + 0.158
    x = start[:, 0].clone()
    y = start[:, 1].clone()
    yaw = start[:, 2].clone()
    u_vel = control[:, 0]
    u_phi = control[:, 1]

    # Constant control has a closed-form constant-curvature endpoint. This is
    # much faster than building an autograd graph with 20-100 Euler substeps per
    # Adam iteration, while the motion validator/visualizer can still sample the
    # curve densely for collision checking.
    yaw_dot = (u_vel / wheelbase) * torch.tan(u_phi)
    duration_t = float(duration)
    end_yaw_raw = yaw + yaw_dot * duration_t
    eps = torch.tensor(1e-6, dtype=start.dtype, device=start.device)
    yaw_dot_safe = torch.where(
        torch.abs(yaw_dot) < eps,
        torch.where(yaw_dot < 0.0, -eps, eps),
        yaw_dot,
    )
    x_curve = x + (u_vel / yaw_dot_safe) * (torch.sin(end_yaw_raw) - torch.sin(yaw))
    y_curve = y - (u_vel / yaw_dot_safe) * (torch.cos(end_yaw_raw) - torch.cos(yaw))
    x_straight = x + u_vel * torch.cos(yaw) * duration_t
    y_straight = y + u_vel * torch.sin(yaw) * duration_t
    straight = torch.abs(yaw_dot) < eps
    new_x = torch.where(straight, x_straight, x_curve)
    new_y = torch.where(straight, y_straight, y_curve)
    new_yaw = torch.remainder(end_yaw_raw + torch.pi, 2 * torch.pi) - torch.pi
    return torch.stack([new_x, new_y, new_yaw], dim=1)


def _double_integrator_dynamics_torch(
    start: torch.Tensor,
    control: torch.Tensor,
    *,
    duration: float,
    integration_step_size: float | None = None,
) -> torch.Tensor:
    pos = start[:, :3]
    vel = start[:, 3:6]
    acc = control[:, :3]
    new_pos = pos + vel * duration + 0.5 * acc * (duration**2)
    new_vel = vel + acc * duration
    return torch.cat([new_pos, new_vel], dim=1)


def _control_bounds(system):
    canonical_name = {
        "simple_car": "kinematic_car",
        "pushing": "pushing_object",
    }.get(system, system)
    try:
        return get_system(canonical_name).control_bounds
    except Exception:
        return None


def _clamp_controls_(controls: torch.Tensor, bounds):
    if bounds is None:
        return
    with torch.no_grad():
        for i, (low, high) in enumerate(bounds):
            controls[:, i].clamp_(float(low), float(high))


def _run_dynamic_optimizer(
    system,
    nextState,
    childrenStatesArray,
    childrenControlsArray,
    controlDuration,
    numStates=1000,
    posSTD=0.003,
    rotSTD=0.05,
    velSTD=None,
    numSteps=25,
    learningRate=0.05,
    integrationStepSize=None,
    maxWallTime=None,
    stopEvent=None,
    partialResultCallback=None,
):
    try:
        deadline = (
            time.monotonic() + max(float(maxWallTime), 0.0)
            if maxWallTime is not None
            else None
        )
        wall_budget = None if maxWallTime is None else max(float(maxWallTime), 0.0)
        device_info = optimizer_device_info()
        device = torch.device(str(device_info["device"]))
        target_states_np = np.array(childrenStatesArray, dtype=float)
        children_controls_np = np.array(childrenControlsArray, dtype=float)
        if target_states_np.size == 0 or children_controls_np.size == 0:
            return None
        if target_states_np.ndim == 1:
            target_states_np = target_states_np.reshape(1, -1)
        if children_controls_np.ndim == 1:
            children_controls_np = children_controls_np.reshape(1, -1)

        num_children = min(len(target_states_np), len(children_controls_np))
        if num_children == 0:
            return None
        target_states_np = target_states_np[:num_children]
        children_controls_np = children_controls_np[:num_children]

        if integrationStepSize is None or float(integrationStepSize) <= 0.0:
            integration_steps = 1
        else:
            integration_steps = max(1, int(np.ceil(float(controlDuration) / float(integrationStepSize))))
        dynamics_work_steps = 1 if system == "kinematic_car" else integration_steps

        requested_num_states = int(numStates)
        requested_num_steps = int(numSteps)
        requested_rows = max(1, requested_num_states * num_children)

        # AURA only has one control-duration window to use this result. If the
        # requested batch is too large, the first Adam step can exceed the whole
        # window and the caller receives no optimizer result at all. Cap the
        # child-major batch by rough dynamics work so the optimizer behaves as an
        # anytime module instead of a best-effort thread that may never report.
        using_cuda = device.type == "cuda"
        if wall_budget is None:
            max_rows = 60000
            max_dynamics_work = 1200000
            max_steps = requested_num_steps
        elif using_cuda:
            # On GPU, keep the requested optimizer much closer to the theoretical
            # AURA method. The wall-clock deadline still makes the result anytime.
            max_rows = int(np.clip(120000 * max(wall_budget, 0.25), 30000, 500000))
            max_dynamics_work = int(
                np.clip(5000000 * max(wall_budget, 0.25), 1000000, 50000000)
            )
            max_steps = requested_num_steps
        else:
            max_rows = int(np.clip(8000 * max(wall_budget, 0.25), 1500, 40000))
            max_dynamics_work = int(
                np.clip(140000 * max(wall_budget, 0.25), 40000, 900000)
            )
            max_steps = int(np.clip(25 * max(wall_budget, 0.25), 6, 120))

        rows_by_work = max(1, max_dynamics_work // max(1, dynamics_work_steps))
        effective_rows = min(requested_rows, max_rows, rows_by_work)
        effective_num_states = max(1, effective_rows // num_children)
        effective_num_steps = max(1, min(requested_num_steps, max_steps))

        sampled_states_np = np.array(
            _sample_random_states(
                system,
                nextState,
                effective_num_states,
                posSTD,
                rotSTD,
                velSTD,
            ),
            dtype=float,
        )
        if len(sampled_states_np) == 0:
            return None

        # Child-major ordering keeps tensors compatible with AURA.pick_next_control:
        # child 0 sampled starts, child 1 sampled starts, ...
        repeated_controls = np.repeat(children_controls_np, effective_num_states, axis=0)
        repeated_start_states = np.tile(sampled_states_np, (num_children, 1))
        repeated_target_states = np.repeat(target_states_np, effective_num_states, axis=0)

        startStates = torch.tensor(
            repeated_start_states, requires_grad=False, dtype=torch.float32, device=device
        )
        targetStates = torch.tensor(
            repeated_target_states, requires_grad=False, dtype=torch.float32, device=device
        )
        initialControls = torch.tensor(
            repeated_controls, requires_grad=True, dtype=torch.float32, device=device
        )

        optimizer = torch.optim.Adam([initialControls], lr=learningRate)
        loss_history_tensors: list[torch.Tensor] = []
        bounds = _control_bounds(system)
        start_states_cpu = None
        target_states_cpu = None

        dynamics = (
            _kinematic_car_dynamics_torch
            if system in ("kinematic_car", "simple_car")
            else _double_integrator_dynamics_torch
        )

        def _make_result(*, partial: bool, timed_out_flag: bool, stopped_flag: bool) -> dict:
            nonlocal start_states_cpu, target_states_cpu
            if using_cuda:
                torch.cuda.synchronize(device)
            if start_states_cpu is None:
                start_states_cpu = startStates.detach().cpu()
            if target_states_cpu is None:
                target_states_cpu = targetStates.detach().cpu()
            loss_history = (
                torch.stack(loss_history_tensors).detach().cpu().numpy().astype(float).tolist()
                if loss_history_tensors
                else []
            )
            return {
                "optimized_controls": initialControls.detach().cpu(),
                "start_states": start_states_cpu,
                "target_states": target_states_cpu,
                "optimization_success": True,
                "loss_history": loss_history,
                "initial_loss": loss_history[0] if loss_history else None,
                "final_loss": loss_history[-1] if loss_history else None,
                "timed_out": timed_out_flag,
                "stopped": stopped_flag,
                "partial_result": partial,
                "steps_completed": len(loss_history),
                "requested_num_states": requested_num_states,
                "effective_num_states": effective_num_states,
                "requested_num_steps": requested_num_steps,
                "effective_num_steps": effective_num_steps,
                "num_children": num_children,
                "integration_steps": integration_steps,
                "dynamics_work_steps": dynamics_work_steps,
                "wall_time_budget": wall_budget,
                "device": str(device),
                "cuda_available": bool(device_info.get("cuda_available", False)),
                "device_name": str(device_info.get("device_name", "")),
            }

        def _publish_partial(*, timed_out_flag: bool = False, stopped_flag: bool = False) -> None:
            if partialResultCallback is None or not loss_history_tensors:
                return
            partial = _make_result(
                partial=True,
                timed_out_flag=timed_out_flag,
                stopped_flag=stopped_flag,
            )
            partialResultCallback(partial)

        timed_out = False
        sync_every = 8 if using_cuda else 1
        for step_idx in range(effective_num_steps):
            if stopEvent is not None and stopEvent.is_set():
                timed_out = True
                break
            if deadline is not None and step_idx > 0 and time.monotonic() >= deadline:
                timed_out = True
                break
            optimizer.zero_grad(set_to_none=True)
            predictedStates = dynamics(
                startStates,
                initialControls,
                duration=controlDuration,
                integration_step_size=integrationStepSize,
            )
            current_loss = torch.nn.functional.mse_loss(predictedStates, targetStates)
            current_loss.backward()
            optimizer.step()
            _clamp_controls_(initialControls, bounds)
            loss_history_tensors.append(current_loss.detach())
            publish_now = (
                partialResultCallback is not None
                and (
                    step_idx == 0
                    or (step_idx + 1) % sync_every == 0
                    or step_idx + 1 == effective_num_steps
                )
            )
            if publish_now:
                _publish_partial()
            elif using_cuda and (step_idx + 1) % sync_every == 0:
                torch.cuda.synchronize(device)

            if stopEvent is not None and stopEvent.is_set():
                timed_out = True
                _publish_partial(timed_out_flag=True, stopped_flag=True)
                break
            if deadline is not None and (publish_now or (step_idx + 1) % sync_every == 0):
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    _publish_partial(timed_out_flag=True, stopped_flag=False)
                    break

        return _make_result(
            partial=False,
            timed_out_flag=timed_out,
            stopped_flag=bool(stopEvent is not None and stopEvent.is_set()),
        )
    except Exception as e:
        log(f"[ERROR] in dynamic optimizer: {e}", "error")
        import traceback

        traceback.print_exc()
        return None


def runOptimizer(
    system,
    nextState,
    childrenStatesArray,
    childrenControlsArray,
    optModel,
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
    velSTD=None,
    originalControl=None,
    controlDuration=0.1,
    integrationStepSize=None,
    numSteps=25,
    learningRate=0.05,
    maxWallTime=None,
    stopEvent=None,
    partialResultCallback=None,
):
    try:
        system_key = {
            "kinematic_car": "kinematic_car",
            "pushing_object": "pushing_object",
            "double_integrator": "double_integrator",
            "simple_car": "kinematic_car",
            "pushing": "pushing_object",
        }.get(system, system)

        if system_key in ("kinematic_car", "double_integrator"):
            return _run_dynamic_optimizer(
                system=system_key,
                nextState=nextState,
                childrenStatesArray=childrenStatesArray,
                childrenControlsArray=childrenControlsArray,
                controlDuration=controlDuration,
                numStates=numStates,
                posSTD=posSTD,
                rotSTD=rotSTD,
                velSTD=velSTD,
                numSteps=numSteps,
                learningRate=learningRate,
                integrationStepSize=integrationStepSize,
                maxWallTime=maxWallTime,
                stopEvent=stopEvent,
                partialResultCallback=partialResultCallback,
            )

        sampledStates_raw = np.array(
            _sample_random_states(system_key, nextState, numStates, posSTD, rotSTD, velSTD)
        )

        if len(sampledStates_raw) == 0:
            log("[WARNING] No sampled states generated. Cannot run optimizer.", "warning")
            return None

        sampledStates = [SE2Pose(state[:2], state[2]) for state in sampledStates_raw]

        childrenStates_raw = np.array(childrenStatesArray)
        if len(childrenStates_raw) == 0:
            log("[WARNING] No children states found. Cannot run optimizer.", "warning")
            return None

        optimizer_childrenStates = [SE2Pose(state[:2], state[2]) for state in childrenStates_raw]
        numChildren = len(optimizer_childrenStates)
        if numChildren == 0:
            log("[WARNING] No children states found. Cannot run optimizer.", "warning")
            return None

        sampled_inverts = [s.invert for s in sampledStates]
        if len(sampled_inverts) == 0:
            log("[WARNING] No sampled state inverses. Cannot run optimizer.", "warning")
            return None

        relativePoses_matrix = [
            [inv @ c for c in optimizer_childrenStates] for inv in sampled_inverts
        ]
        if len(relativePoses_matrix) == 0:
            log("[WARNING] relativePoses_matrix is empty. Cannot run optimizer.", "warning")
            return None
        if any(len(row) == 0 for row in relativePoses_matrix):
            log("[WARNING] relativePoses_matrix has empty rows. Cannot run optimizer.", "warning")
            return None

        relativePosesFlat = np.array(
            [
                [pose.position[0], pose.position[1], pose.euler[2]]
                for row in relativePoses_matrix
                for pose in row
            ]
        )
        if relativePosesFlat.size == 0 or len(relativePosesFlat) == 0:
            log("[WARNING] relativePosesFlat is empty. Cannot run optimizer.", "warning")
            log(
                f"[DEBUG] numStates: {numStates}, numChildren: {numChildren}, sampledStates_raw shape: {sampledStates_raw.shape}, childrenStates_raw shape: {childrenStates_raw.shape}",
                "warning",
            )
            log(
                f"[DEBUG] relativePoses_matrix length: {len(relativePoses_matrix)}, first row length: {len(relativePoses_matrix[0]) if relativePoses_matrix else 0}",
                "warning",
            )
            return None

        initialGuessControlsFlat = []
        if originalControl is not None:
            originalControl_np = (
                originalControl.cpu().numpy()
                if hasattr(originalControl, "cpu")
                else np.array(originalControl)
            )
            print(
                f"[DEBUG] [OPTIMIZER] Using original control as initial guess: {originalControl_np}"
            )
            for _ in range(numStates):
                for _ in range(numChildren):
                    initialGuessControlsFlat.append(originalControl_np)
        else:
            print(
                "[DEBUG] [OPTIMIZER] No original control provided, using children controls as initial guess"
            )
            for _ in range(numStates):
                for j in range(numChildren):
                    if j < len(childrenControlsArray):
                        initialGuessControlsFlat.append(childrenControlsArray[j])
                    else:
                        initialGuessControlsFlat.append(
                            childrenControlsArray[0]
                            if len(childrenControlsArray) > 0
                            else [0.0, 0.0, 0.0]
                        )

        initialGuessControlsFlat = np.array(initialGuessControlsFlat)
        # Pushing controls use normalized face ids in the learned model and
        # MuJoCo bridge: 0.0, 0.25, 0.5, 0.75. The real-world path generator
        # converts this normalized value to radians only at execution time.
        x_min = np.array([0.0, -0.4, 0.0])
        x_max = np.array([0.75, 0.4, 0.3])

        if optModel is None:
            raise ValueError("optModel is None for pushing system. Cannot run optimizer.")

        optimizedControlsFlat, loss = optModel.predict(
            relativePosesFlat, initialGuessControlsFlat, x_min=x_min, x_max=x_max, plot=False
        )
        # AURA's optimizer is intended as a local correction around the branch
        # control. For pushing, unconstrained learned-model inversion often
        # over-corrects by increasing push distance, which transfers poorly to
        # MuJoCo/real contact and can overshoot the object. Keep side offset and
        # distance in a small trust region around the original control.
        original_guess_flat = np.asarray(initialGuessControlsFlat, dtype=float)
        if original_guess_flat.shape == optimizedControlsFlat.shape and optimizedControlsFlat.shape[-1] >= 3:
            side_trust = 0.08
            distance_trust = 0.010
            optimizedControlsFlat[:, 1] = np.clip(
                optimizedControlsFlat[:, 1],
                original_guess_flat[:, 1] - side_trust,
                original_guess_flat[:, 1] + side_trust,
            )
            optimizedControlsFlat[:, 2] = np.clip(
                optimizedControlsFlat[:, 2],
                np.maximum(0.0, original_guess_flat[:, 2] - distance_trust),
                np.minimum(0.12, original_guess_flat[:, 2] + distance_trust),
            )
        print(f"[DEBUG] [OPTIMIZER] Optimization completed with final loss: {loss:.6f}")

        try:
            if hasattr(optModel, "loss_history"):
                loss_history = optModel.loss_history
                print(
                    f"[DEBUG] [OPTIMIZER] Loss history (last 5): {loss_history[-5:] if len(loss_history) > 5 else loss_history}"
                )
        except Exception:
            pass

        control_dim = optimizedControlsFlat.shape[1] if len(optimizedControlsFlat.shape) > 1 else 1
        optimizedControls = optimizedControlsFlat.reshape(numStates, numChildren, control_dim)

        startStates_array = np.array(
            [[pose.position[0], pose.position[1], pose.euler[2]] for pose in sampledStates]
        )
        targetStates_array = np.array(
            [
                [pose.position[0], pose.position[1], pose.euler[2]]
                for pose in optimizer_childrenStates
            ]
        )

        # The pushing face/contact side is discrete. Optimizing it as a
        # continuous variable and snapping afterward can silently switch the
        # robot to a different side of the object, which is not a local control
        # correction anymore. Keep the branch face fixed and only tune the
        # continuous side offset and push distance.
        children_controls_for_faces = np.asarray(childrenControlsArray, dtype=float)
        if children_controls_for_faces.ndim == 1:
            children_controls_for_faces = children_controls_for_faces.reshape(1, -1)
        branch_faces = children_controls_for_faces[:numChildren, 0].reshape(1, numChildren)
        optimizedControls[:, :, 0] = branch_faces

        # Side offset is relative to the contacted edge, matching
        # simulation/collect_push_data.py. The path generator converts this to meters.
        optimizedControls[:, :, 1] = np.clip(optimizedControls[:, :, 1], -0.4, 0.4)

        optimizedControls_reshaped = optimizedControls.transpose(1, 0, 2)
        optimizedControls_flat = optimizedControls_reshaped.reshape(
            numChildren * numStates, control_dim
        )

        startStates_flat = np.tile(startStates_array, (numChildren, 1))
        targetStates_flat = np.repeat(targetStates_array, numStates, axis=0)

        optimizedControls_torch = torch.tensor(optimizedControls_flat, dtype=torch.float64)
        startStates_torch = torch.tensor(startStates_flat, dtype=torch.float64)
        targetStates_torch = torch.tensor(targetStates_flat, dtype=torch.float64)

        result = {
            "optimized_controls": optimizedControls_torch,
            "start_states": startStates_torch,
            "target_states": targetStates_torch,
            "optimization_success": True,
            "initial_loss": None,
            "final_loss": loss,
        }

        try:
            if hasattr(optModel, "loss_history"):
                result["loss_history"] = optModel.loss_history
                lh = optModel.loss_history
                if lh and result.get("initial_loss") is None:
                    result["initial_loss"] = lh[0]
            if hasattr(optModel, "learning_rate_history"):
                result["learning_rate_history"] = optModel.learning_rate_history
        except Exception:
            pass

        return result

    except Exception as e:
        log(f"[ERROR] in runOptimizer: {e}", "error")
        import traceback

        traceback.print_exc()
        return None
