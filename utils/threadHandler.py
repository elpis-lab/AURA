import torch
import threading
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple

from geometry.pose import SE2Pose

from utils.utils import log
from utils.solutionsHandler import getSolutionsInfo
from utils.childrenHandler import sampleRandomState

from planning.propagators import (
    dublinsAirplaneDynamics,
    dublinsAirplaneDynamicsTorch,
    carDynamics,
    carDynamicsTorch,
    pushingDynamics,
    pushingDynamicsTorch,
)


def executeWaypoints(client, pos_waypoints, resultContainer):
    result = client.execute("execute_waypoints", pos_waypoints)
    resultContainer["result"] = result
    resultContainer["completed"] = True


def createExecuteThread(client, pos_waypoints):
    resultContainer = {"result": None, "completed": False}
    print(f"[INFO] Execute thread created")
    thread = threading.Thread(
        target=executeWaypoints,
        args=(client, pos_waypoints, resultContainer),
    )
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def simulatedExecution(system, state, control, totalTime, dt, resultContainer, client):
    try:
        if system == "dublin_airplane":
            # Convert state to numpy if it's a tensor
            if hasattr(state, "cpu"):
                state_np = state.detach().cpu().numpy()
            elif isinstance(state, (list, tuple)):
                state_np = np.array(state)
            else:
                state_np = state

            # Convert control to numpy if it's a tensor
            if hasattr(control, "cpu"):
                control_np = control.detach().cpu().numpy()
            elif isinstance(control, (list, tuple)):
                control_np = np.array(control)
            else:
                control_np = control

            result = dublinsAirplaneDynamics(state_np, control_np, totalTime)

        elif system == "simple_car":
            # Convert state to numpy if it's a tensor
            if hasattr(state, "cpu"):
                current_state = state.detach().cpu().numpy()
            elif isinstance(state, np.ndarray):
                current_state = state.copy()
            elif isinstance(state, (list, tuple)):
                current_state = np.array(state)
            else:
                current_state = np.array(state)

            # Create a local copy to avoid modifying the original parameter
            current_state = current_state.copy()

            # Convert control to numpy if it's a tensor
            if hasattr(control, "cpu"):
                control_np = control.detach().cpu().numpy()
            else:
                control_np = control

            # Validate inputs
            if totalTime <= 0 or dt <= 0:
                raise ValueError(f"Invalid time parameters: totalTime={totalTime}, dt={dt}")

            if current_state is None or control_np is None:
                raise ValueError(
                    f"Invalid state or control: state={current_state}, control={control_np}"
                )

            # Execute the control for the full duration in one call (similar to pushing)
            # The execute_segment function handles the full duration execution on the simulation side
            try:
                mujoco_state = client.execute(
                    "execute_segment", params=[control_np, totalTime]
                )

                if mujoco_state is None:
                    raise Exception("MuJoCo execution failed to return a current state.")

                # Extract state (first 3 elements for SE2: [x, y, theta])
                # Note: simple_car get_state() returns [x, y, theta, v], so we take [:3]
                current_state = mujoco_state[:3]  # Update local copy

            except Exception as e:
                print(f"[ERROR] MuJoCo execution failed within thread: {e}")
                import traceback

                traceback.print_exc()
                raise e

            result = current_state  # Return final result

        elif system == "pushing":
            # Convert state to numpy if it's a tensor
            if hasattr(state, "cpu"):
                current_state = state.detach().cpu().numpy()
            elif isinstance(state, np.ndarray):
                current_state = state.copy()
            elif isinstance(state, (list, tuple)):
                current_state = np.array(state)
            else:
                current_state = np.array(state)

            # Create a local copy to avoid modifying the original parameter
            current_state = current_state.copy()

            # Convert control to numpy if it's a tensor
            if hasattr(control, "cpu"):
                control_np = control.detach().cpu().numpy()
            else:
                control_np = control

            # Validate inputs
            if totalTime <= 0 or dt <= 0:
                raise ValueError(f"Invalid time parameters: totalTime={totalTime}, dt={dt}")

            if current_state is None or control_np is None:
                raise ValueError(
                    f"Invalid state or control: state={current_state}, control={control_np}"
                )

            num_steps = int(totalTime / dt)

            # Use default object shape for pushing (cracker box)
            # This matches the default in pushingDynamics function
            object_shape = np.array([0.1628, 0.2139, 0.0676])

            # print(f"[DEBUG] Executing pushing dynamics for {num_steps} steps")
            # print(f"[DEBUG] Start state: {current_state}")
            # print(f"[DEBUG] Control: {control_np}")

            for i in range(num_steps):
                # We don't need step-by-step dynamics here if we are using the bridge
                # result = pushingDynamics(current_state, control_np, dt, object_shape=object_shape)

                # if result is None:
                #     raise RuntimeError(f"pushingDynamics returned None at step {i}")

                # # Validate result
                # if not isinstance(result, (np.ndarray, list)) or len(result) != 3:
                #     raise RuntimeError(
                #         f"pushingDynamics returned invalid result at step {i}: {result} (type: {type(result)})"
                #     )

                # current_state = object_pose / control_np = push_params / totalTime = duration 2.0 sec

                try:
                    if i == 0:  # Only call bridge once
                        # ws_path = client.execute("generate_ws_path", params=[control_np, totalTime])
                        # # This executes the whole trajectory on the server side
                        # way_point_0 = client.execute(
                        #     "get_final_waypoints", params=[control_np, totalTime, ws_path]
                        # )
                        # mujoco_state = client.execute("get_state")

                        mujoco_state = client.execute(
                            "execute_segment", params=[control_np, totalTime]
                        )

                        if mujoco_state is None:
                            raise Exception("MuJoCo execution failed to return a current state.")

                        # print(f"[DEBUG] MuJoCo returned state: {mujoco_state}")
                        current_state = mujoco_state  # Update local copy
                        break  # Exit loop after one bridge call as it handles the full duration

                except Exception as e:
                    print(f"[ERROR] MuJoCo execution failed within thread: {e}")
                    import traceback

                    traceback.print_exc()
                    raise e

                # import ipdb; ipdb.set_trace()

                # current_state = mujoco_state  # Update local copy, not parameter

            result = current_state  # Return final result

        else:
            result = None

        resultContainer["result"] = result
        resultContainer["completed"] = True

    except Exception as e:
        print(f"[ERROR] simulatedExecution failed: {e}")
        import traceback

        traceback.print_exc()
        resultContainer["result"] = None
        resultContainer["completed"] = False


def simulatedExecutionWrapper(system, state, control, totalTime, dt, resultContainer, client):
    """
    Wrapper function for simulatedExecution with additional error handling
    """
    try:
        # Input validation
        if system not in ["dublin_airplane", "simple_car", "pushing"]:
            raise ValueError(f"Invalid system: {system}")

        if state is None:
            raise ValueError("State is None")

        if control is None:
            raise ValueError("Control is None")

        if totalTime <= 0:
            raise ValueError(f"totalTime must be positive, got: {totalTime}")

        if dt <= 0:
            raise ValueError(f"dt must be positive, got: {dt}")

        if totalTime < dt:
            raise ValueError(f"totalTime ({totalTime}) must be >= dt ({dt})")

        # Call the actual execution function
        simulatedExecution(system, state, control, totalTime, dt, resultContainer, client)

    except Exception as e:
        print(f"[ERROR] simulatedExecutionWrapper failed: {e}")
        import traceback

        traceback.print_exc()
        resultContainer["result"] = None
        resultContainer["completed"] = False


def createSimulatedExecutionThread(system, state, control, totalTime, dt, client=None):
    resultContainer = {"result": None, "completed": False}
    thread = threading.Thread(
        target=simulatedExecutionWrapper,  # Use the wrapper instead
        args=(system, state, control, totalTime, dt, resultContainer, client),
    )
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def runResolver(ss, replanningTime, resultContainer):
    ss.getPlanner().resolve(replanningTime)
    result = getSolutionsInfo(ss)
    resultContainer["result"] = result
    resultContainer["completed"] = True


def createResolverThread(ss, replanningTime):
    resultContainer = {"result": None, "completed": False}
    # print(f"[INFO] Resolver thread created for replanning time {replanningTime}")
    thread = threading.Thread(target=runResolver, args=(ss, replanningTime, resultContainer))
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


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
    originalControl=None,
):
    # print(f"[INFO] Optimizer thread created")
    # print(f"     - Next state: {nextState[0]:.5f}, {nextState[1]:.5f}, {nextState[2]:.5f}")
    # print(f"     - Number of children states: {len(childrenStatesArray)}")
    # print(f"     - Number of children controls: {len(childrenControlsArray)}")
    # print(f"     - Number of sampled states: {numStates} (maxDistance: {maxDistance})")
    # print(f"     - Max distance: {maxDistance}")
    # print(f"     - Position standard deviation: {posSTD}")
    # print(f"     - Rotation standard deviation: {rotSTD}")

    try:
        sampledStates_raw = np.array(
            sampleRandomState(system, nextState, numStates, posSTD, rotSTD)
        )

        # Check if sampledStates_raw is empty
        if len(sampledStates_raw) == 0:
            log(f"[WARNING] No sampled states generated. Cannot run optimizer.", "warning")
            return None

        sampledStates = [SE2Pose(state[:2], state[2]) for state in sampledStates_raw]

        childrenStates_raw = np.array(childrenStatesArray)

        # Check if childrenStatesArray is empty
        if len(childrenStates_raw) == 0:
            log(f"[WARNING] No children states found. Cannot run optimizer.", "warning")
            return None

        optimizer_childrenStates = [SE2Pose(state[:2], state[2]) for state in childrenStates_raw]
        numChildren = len(optimizer_childrenStates)

        # Check if we have any children states
        if numChildren == 0:
            log(f"[WARNING] No children states found. Cannot run optimizer.", "warning")
            return None

        sampled_inverts = [s.invert for s in sampledStates]

        # Check if sampled_inverts is empty
        if len(sampled_inverts) == 0:
            log(f"[WARNING] No sampled state inverses. Cannot run optimizer.", "warning")
            return None

        # Broadcast: shape will be (numStates, numChildren)
        relativePoses_matrix = [
            [inv @ c for c in optimizer_childrenStates] for inv in sampled_inverts
        ]

        # Check if relativePoses_matrix is empty
        if len(relativePoses_matrix) == 0:
            log(f"[WARNING] relativePoses_matrix is empty. Cannot run optimizer.", "warning")
            return None

        # Check if relativePoses_matrix has any empty rows
        if any(len(row) == 0 for row in relativePoses_matrix):
            log(f"[WARNING] relativePoses_matrix has empty rows. Cannot run optimizer.", "warning")
            return None

        # Flatten into (numStates*numChildren, 3)
        relativePosesFlat = np.array(
            [
                [pose.position[0], pose.position[1], pose.euler[2]]
                for row in relativePoses_matrix
                for pose in row
            ]
        )

        # Check if relativePosesFlat is empty
        if relativePosesFlat.size == 0 or len(relativePosesFlat) == 0:
            log(f"[WARNING] relativePosesFlat is empty. Cannot run optimizer.", "warning")
            log(
                f"[DEBUG] numStates: {numStates}, numChildren: {numChildren}, sampledStates_raw shape: {sampledStates_raw.shape}, childrenStates_raw shape: {childrenStates_raw.shape}",
                "warning",
            )
            log(
                f"[DEBUG] relativePoses_matrix length: {len(relativePoses_matrix)}, first row length: {len(relativePoses_matrix[0]) if relativePoses_matrix else 0}",
                "warning",
            )
            return None

        # Create initial guess controls
        # If originalControl is provided, use it as the initial guess for all samples
        # Otherwise, fall back to using childrenControlsArray
        initialGuessControlsFlat = []
        if originalControl is not None:
            # Convert originalControl to numpy if needed
            originalControl_np = (
                originalControl.cpu().numpy()
                if hasattr(originalControl, "cpu")
                else np.array(originalControl)
            )
            print(
                f"[DEBUG] [OPTIMIZER] Using original control as initial guess: {originalControl_np}"
            )
            # Use the original control as initial guess for all samples
            for i in range(numStates):
                for j in range(numChildren):
                    initialGuessControlsFlat.append(originalControl_np)
        else:
            # Fallback: use children controls (old behavior)
            print(
                f"[DEBUG] [OPTIMIZER] No original control provided, using children controls as initial guess"
            )
            for i in range(numStates):
                for j in range(numChildren):
                    if j < len(childrenControlsArray):
                        initialGuessControlsFlat.append(childrenControlsArray[j])
                    else:
                        # Fallback to the first control if we don't have enough controls
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
            relativePosesFlat,
            initialGuessControlsFlat,
            x_min=x_min,
            x_max=x_max,
            plot=False,
        )
        print(f"[DEBUG] [OPTIMIZER] Optimization completed with final loss: {loss:.6f}")

        # Store loss in result for debugging
        try:
            # Try to get loss history if available
            if hasattr(optModel, "loss_history"):
                loss_history = optModel.loss_history
                print(
                    f"[DEBUG] [OPTIMIZER] Loss history (last 5): {loss_history[-5:] if len(loss_history) > 5 else loss_history}"
                )
        except:
            pass

        control_dim = optimizedControlsFlat.shape[1] if len(optimizedControlsFlat.shape) > 1 else 1
        optimizedControls = optimizedControlsFlat.reshape(numStates, numChildren, control_dim)

        # Convert sampledStates (SE2Pose objects) to numpy arrays
        startStates_array = np.array(
            [[pose.position[0], pose.position[1], pose.euler[2]] for pose in sampledStates]
        )  # (numStates, 3)

        # Convert optimizer_childrenStates (SE2Pose objects) to numpy arrays
        targetStates_array = np.array(
            [
                [pose.position[0], pose.position[1], pose.euler[2]]
                for pose in optimizer_childrenStates
            ]
        )  # (numChildren, 3)

        # Round rotations to nearest multiple of pi/2 before reshaping
        target_rotations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        current_rotations = optimizedControls[:, :, 0]  # (numStates, numChildren)
        diffs = np.abs(
            current_rotations[:, :, np.newaxis] - target_rotations[np.newaxis, np.newaxis, :]
        )  # (numStates, numChildren, 4)
        closest_indices = np.argmin(diffs, axis=2)  # (numStates, numChildren)
        optimizedControls[:, :, 0] = target_rotations[closest_indices]

        # Reshape to match runDynamicOptimizer format: flattened as (numChildren * numStates, dim)
        # The format from runDynamicOptimizer is:
        # - targetStates: [child0, child0, ..., child0 (numStates times), child1, child1, ..., child1, ...]
        #   = repeat_interleave(childrenStatesArray, numStates)
        # - startStates: [state0, state1, ..., stateN, state0, state1, ..., stateN, ...]
        #   = repeat(sampledStates, len(childrenStatesArray))
        # - optimizedControls: same pattern as startStates

        # Current optimizedControls is (numStates, numChildren, control_dim)
        # We need to transpose to (numChildren, numStates, control_dim), then flatten column-wise
        # So: [child0_state0, child0_state1, ..., child0_stateN, child1_state0, child1_state1, ...]
        optimizedControls_reshaped = optimizedControls.transpose(
            1, 0, 2
        )  # (numChildren, numStates, control_dim)
        optimizedControls_flat = optimizedControls_reshaped.reshape(
            numChildren * numStates, control_dim
        )

        # startStates: each sampled state repeated for all children
        # [state0, state1, ..., stateN, state0, state1, ..., stateN, ...]
        # = repeat(sampledStates, numChildren)
        startStates_flat = np.repeat(
            startStates_array, numChildren, axis=0
        )  # (numChildren * numStates, 3)

        # targetStates: each child repeated for all sampled states
        # [child0, child0, ..., child0 (numStates times), child1, child1, ..., child1, ...]
        # = repeat_interleave(childrenStatesArray, numStates)
        targetStates_flat = np.repeat(
            targetStates_array, numStates, axis=0
        )  # (numChildren * numStates, 3)

        # Convert to torch tensors to match runDynamicOptimizer format
        optimizedControls_torch = torch.tensor(optimizedControls_flat, dtype=torch.float64)
        startStates_torch = torch.tensor(startStates_flat, dtype=torch.float64)
        targetStates_torch = torch.tensor(targetStates_flat, dtype=torch.float64)

        # Return in the same format as runDynamicOptimizer
        result = {
            "optimized_controls": optimizedControls_torch,
            "start_states": startStates_torch,
            "target_states": targetStates_torch,
            "optimization_success": True,
            "final_loss": loss,  # Include loss for debugging
        }

        # Try to include loss history if available
        try:
            if hasattr(optModel, "loss_history"):
                result["loss_history"] = optModel.loss_history
            if hasattr(optModel, "learning_rate_history"):
                result["learning_rate_history"] = optModel.learning_rate_history
        except:
            pass

        return result

    except Exception as e:
        log(f"[ERROR] in runOptimizer: {e}", "error")
        import traceback

        traceback.print_exc()
        return None


def controlJacobian(
    dynamics: Callable[..., torch.Tensor],
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    dynamics_kwargs: Optional[Dict[str, Any]] = None,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y, df/du) with shapes:
      y:    (..., state_dim)
      dfdu: (..., state_dim, control_dim)
    """
    if dynamics_kwargs is None:
        dynamics_kwargs = {}

    u = u.clone().requires_grad_(True)

    with torch.enable_grad():
        y = dynamics(x.detach(), u, **dynamics_kwargs)

    # flatten leading batch dims for per-item jacobian clarity
    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, t.shape[-1]) if t.ndim > 1 else t.unsqueeze(0)

    x_flat = _flatten(x.detach())
    u_flat = _flatten(u)
    y_flat = _flatten(y)

    B = x_flat.shape[0]
    state_dim = y_flat.shape[-1]
    from torch.autograd.functional import jacobian

    def per_item(i: int) -> torch.Tensor:
        xi = x_flat[i]
        ui = u_flat[i]

        def g(u_local: torch.Tensor) -> torch.Tensor:
            return dynamics(xi, u_local, **dynamics_kwargs)

        J = jacobian(g, ui, vectorize=True, create_graph=create_graph, strict=True)
        return J.reshape(state_dim, -1)

    Js = [per_item(i) for i in range(B)]
    J_flat = torch.stack(Js, dim=0)  # (B, state_dim, control_dim)

    batch_shape = y.shape[:-1]
    control_dim = J_flat.shape[-1]
    dfdu = J_flat.reshape(*batch_shape, state_dim, control_dim)
    return y, dfdu


def clampControls(controls, controlBounds):
    controls[:, 0] = torch.clamp(controls[:, 0], controlBounds[0], controlBounds[1])
    controls[:, 1] = torch.clamp(controls[:, 1], controlBounds[2], controlBounds[3])


def runDynamicOptimizer(
    system,
    nextState,
    childrenStatesArray,
    childrenControlsArray,
    controlDuration,
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
    numSteps=10,
    learningRate=0.1,
    plateau_factor=0.5,
    plateau_patience=2,
    plateau_min_lr=1e-6,
):
    """
    Optimizer specifically for dynamic systems like Dublin airplane using PyTorch-based optimization.
    """
    # print(f"[INFO] Dynamic Optimizer thread created")
    # print(f"     - Next state: {nextState[0]:.5f}, {nextState[1]:.5f}, {nextState[2]:.5f}")
    # print(f"     - Number of children states: {len(childrenStatesArray)}")
    # print(f"     - Number of children controls: {len(childrenControlsArray)}")
    # print(f"     - Number of sampled states: {numStates} (maxDistance: {maxDistance})")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sampledStates = torch.tensor(
            np.array(sampleRandomState(system, nextState, numStates, posSTD, rotSTD)),
            dtype=torch.float64,
            device=device,
        )

        targetStates = torch.tensor(
            np.array(childrenStatesArray), dtype=torch.float64, device=device
        ).repeat_interleave(numStates, dim=0)

        base_controls = np.array(childrenControlsArray)
        repeated_controls = np.repeat(base_controls, numStates, axis=0)

        initialControls = torch.tensor(
            repeated_controls, requires_grad=True, dtype=torch.float64, device=device
        )

        sampled_states_np = sampleRandomState(system, nextState, numStates, posSTD, rotSTD)
        repeated_states_np = np.repeat(sampled_states_np, len(childrenStatesArray), axis=0)
        startStates = torch.tensor(repeated_states_np, dtype=torch.float64, device=device)

        test_loss = torch.sum(initialControls**2)
        test_loss.backward()

        initialControls.grad.zero_()

        def compute_loss(pred, target):
            return torch.nn.functional.mse_loss(pred, target)

        optimizer = torch.optim.Adam([initialControls], lr=learningRate)

        # Add plateau regulator to automatically reduce learning rate when loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Monitor loss (minimize)
            factor=plateau_factor,  # Reduce LR by half when plateauing
            patience=plateau_patience,  # Wait 2 steps before reducing LR
            min_lr=plateau_min_lr,  # Minimum learning rate
        )

        # Track loss history for plateau detection
        loss_history = []
        lr_history = []

        for step in range(numSteps):
            optimizer.zero_grad(set_to_none=True)

            if system == "dublin_airplane":
                predictedStates = dublinsAirplaneDynamicsTorch(
                    startStates, initialControls, duration=controlDuration
                )
            elif system == "simple_car":
                predictedStates = carDynamicsTorch(
                    startStates, initialControls, duration=controlDuration
                )
            elif system == "pushing":
                predictedStates = pushingDynamicsTorch(
                    startStates, initialControls, duration=controlDuration
                )
            else:
                raise ValueError(f"Invalid system: {system}")

            current_loss = compute_loss(predictedStates, targetStates)
            current_loss.backward()
            optimizer.step()

            # Update scheduler with current loss
            scheduler.step(current_loss)
            loss_history.append(current_loss.item())
            lr_history.append(optimizer.param_groups[0]["lr"])

            # Print progress every few steps
            if step % max(1, numSteps // 5) == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                # print(
                #     f"     - Step {step}: Loss = {current_loss.item():.6f}, LR = {current_lr:.6f}"
                # )

        print(f"     - Final Loss: {current_loss}")
        # print(f"     - Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(
            # f"     - Loss history: {[f'{l:.6f}' for l in loss_history[:5]]}..."
        )  # Show first 5 losses

        # Create result dictionary
        result = {
            "optimized_controls": initialControls,
            "start_states": startStates,
            "target_states": targetStates,
            "optimization_success": True,
            "loss_history": loss_history,
            "learning_rate_history": lr_history,
            "step_history": list(range(numSteps)),
        }

        return result

    except Exception as e:
        print(f"[ERROR] Error in dynamic optimizer: {e}")
        import traceback

        traceback.print_exc()
        return None


def createOptimizerThread(
    system,
    nextState,
    childrenStates,
    childrenControls,
    controlDuration,
    optModel,
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
    numSteps=10,
    learningRate=0.1,
    plateau_factor=0.5,
    plateau_patience=2,
    plateau_min_lr=1e-6,
    originalControl=None,
):
    resultContainer = {"result": None, "completed": False}

    # Choose optimizer based on system type
    if system == "simple_car":

        def optimizer_wrapper():
            try:
                result = runDynamicOptimizer(
                    system,
                    nextState,
                    childrenStates,
                    childrenControls,
                    controlDuration,
                    numStates,
                    maxDistance,
                    posSTD,
                    rotSTD,
                    numSteps,
                    learningRate,
                    plateau_factor,
                    plateau_patience,
                    plateau_min_lr,
                )
                resultContainer["result"] = result
                resultContainer["completed"] = True
                print(f"[INFO] Simple car optimizer completed with result: {result is not None}")
            except Exception as e:
                print(f"[ERROR] Error in simple car optimizer: {e}")
                import traceback

                traceback.print_exc()
                resultContainer["result"] = None
                resultContainer["completed"] = True

        thread = threading.Thread(target=optimizer_wrapper)

    elif system == "dublin_airplane":

        def optimizer_wrapper():
            try:
                result = runDynamicOptimizer(
                    system,
                    nextState,
                    childrenStates,
                    childrenControls,
                    controlDuration,
                    numStates,
                    maxDistance,
                    posSTD,
                    rotSTD,
                    numSteps,
                    learningRate,
                    plateau_factor,
                    plateau_patience,
                    plateau_min_lr,
                )
                resultContainer["result"] = result
                resultContainer["completed"] = True
                print(
                    f"[INFO] Dublin airplane optimizer completed with result: {result is not None}"
                )
            except Exception as e:
                print(f"[ERROR] Error in dublin airplane optimizer: {e}")
                import traceback

                traceback.print_exc()
                resultContainer["result"] = result
                resultContainer["completed"] = True

        thread = threading.Thread(target=optimizer_wrapper)

    elif system == "pushing":

        def optimizer_wrapper():
            try:
                result = runOptimizer(
                    system,
                    nextState,
                    childrenStates,
                    childrenControls,
                    optModel,
                    numStates,
                    maxDistance,
                    posSTD,
                    rotSTD,
                    originalControl=originalControl,
                )
                resultContainer["result"] = result
                resultContainer["completed"] = True
                print(f"[INFO] Pushing optimizer completed with result: {result is not None}")
            except Exception as e:
                print(f"[ERROR] Error in pushing optimizer: {e}")
                import traceback

                traceback.print_exc()
                resultContainer["result"] = None
                resultContainer["completed"] = True

        thread = threading.Thread(target=optimizer_wrapper)

    else:
        print(f"[ERROR] Unknown system type: {system}")
        resultContainer["result"] = None
        resultContainer["completed"] = True

    thread.daemon = True
    thread.resultContainer = resultContainer

    # Debug: Print initial state
    # print(f"[DEBUG] Created optimizer thread for system: {system}")
    # print(f"[DEBUG] Initial resultContainer: {resultContainer}")

    return thread
