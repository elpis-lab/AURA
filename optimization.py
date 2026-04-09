from __future__ import annotations

import numpy as np
import torch

from geometry.pose import SE2Pose
from systems import get_system
from utils.childrenHandler import sampleRandomState
from utils.utils import log


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
    start: torch.Tensor, control: torch.Tensor, *, duration: float
) -> torch.Tensor:
    wheelbase = 0.1385 + 0.158
    x = start[:, 0]
    y = start[:, 1]
    yaw = start[:, 2]
    u_vel = control[:, 0]
    u_phi = control[:, 1]

    x_dot = u_vel * torch.cos(yaw)
    y_dot = u_vel * torch.sin(yaw)
    yaw_dot = (u_vel / wheelbase) * torch.tan(u_phi)

    new_x = x + x_dot * duration
    new_y = y + y_dot * duration
    new_yaw = torch.remainder(yaw + yaw_dot * duration + torch.pi, 2 * torch.pi) - torch.pi
    return torch.stack([new_x, new_y, new_yaw], dim=1)


def _double_integrator_dynamics_torch(
    start: torch.Tensor, control: torch.Tensor, *, duration: float
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
):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sampled_states_np = np.array(
            _sample_random_states(system, nextState, numStates, posSTD, rotSTD, velSTD), dtype=float
        )
        if len(sampled_states_np) == 0:
            return None

        target_states_np = np.array(childrenStatesArray, dtype=float)
        repeated_controls = np.repeat(
            np.array(childrenControlsArray, dtype=float), numStates, axis=0
        )
        repeated_start_states = np.repeat(sampled_states_np, len(childrenStatesArray), axis=0)
        repeated_target_states = np.repeat(target_states_np, numStates, axis=0)

        startStates = torch.tensor(
            repeated_start_states, requires_grad=False, dtype=torch.float64, device=device
        )
        targetStates = torch.tensor(
            repeated_target_states, requires_grad=False, dtype=torch.float64, device=device
        )
        initialControls = torch.tensor(
            repeated_controls, requires_grad=True, dtype=torch.float64, device=device
        )

        optimizer = torch.optim.Adam([initialControls], lr=learningRate)
        loss_history = []
        bounds = _control_bounds(system)

        dynamics = (
            _kinematic_car_dynamics_torch
            if system in ("kinematic_car", "simple_car")
            else _double_integrator_dynamics_torch
        )

        for _ in range(numSteps):
            optimizer.zero_grad(set_to_none=True)
            predictedStates = dynamics(startStates, initialControls, duration=controlDuration)
            current_loss = torch.nn.functional.mse_loss(predictedStates, targetStates)
            current_loss.backward()
            optimizer.step()
            _clamp_controls_(initialControls, bounds)
            loss_history.append(float(current_loss.item()))

        return {
            "optimized_controls": initialControls.detach().cpu(),
            "start_states": startStates.detach().cpu(),
            "target_states": targetStates.detach().cpu(),
            "optimization_success": True,
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
        }
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
        x_min = np.array([0.0, -0.1, 0.0])
        x_max = np.array([2 * np.pi, 0.1, 0.3])

        if optModel is None:
            raise ValueError("optModel is None for pushing system. Cannot run optimizer.")

        optimizedControlsFlat, loss = optModel.predict(
            relativePosesFlat, initialGuessControlsFlat, x_min=x_min, x_max=x_max, plot=False
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

        target_rotations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        current_rotations = optimizedControls[:, :, 0]
        diffs = np.abs(
            current_rotations[:, :, np.newaxis] - target_rotations[np.newaxis, np.newaxis, :]
        )
        closest_indices = np.argmin(diffs, axis=2)
        optimizedControls[:, :, 0] = target_rotations[closest_indices]

        optimizedControls_reshaped = optimizedControls.transpose(1, 0, 2)
        optimizedControls_flat = optimizedControls_reshaped.reshape(
            numChildren * numStates, control_dim
        )

        startStates_flat = np.repeat(startStates_array, numChildren, axis=0)
        targetStates_flat = np.repeat(targetStates_array, numStates, axis=0)

        optimizedControls_torch = torch.tensor(optimizedControls_flat, dtype=torch.float64)
        startStates_torch = torch.tensor(startStates_flat, dtype=torch.float64)
        targetStates_torch = torch.tensor(targetStates_flat, dtype=torch.float64)

        result = {
            "optimized_controls": optimizedControls_torch,
            "start_states": startStates_torch,
            "target_states": targetStates_torch,
            "optimization_success": True,
            "final_loss": loss,
        }

        try:
            if hasattr(optModel, "loss_history"):
                result["loss_history"] = optModel.loss_history
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
