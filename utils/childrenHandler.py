"""Planner-tree child extraction and local state sampling."""

from __future__ import annotations

import numpy as np
from ompl import control as oc
from ompl import util as ou

from utils.control_duration import ControlEdge, duration_seconds_to_steps
from utils.utils import arrayDistance, isStateEqual, log, state2list


DEFAULT_NEAREST_VERTEX_MAX_DIST = 0.25


def getOutgoingEdgeIndices(planner_data, vertex_index: int) -> list[int]:
    """Support both current nanobind and legacy Boost.Python PlannerData APIs."""
    try:
        return [int(value) for value in planner_data.getEdges(int(vertex_index))]
    except TypeError:
        child_vertex_indices = ou.vectorUint()
        planner_data.getEdges(int(vertex_index), child_vertex_indices)
        return [int(value) for value in child_vertex_indices]


def getChildEdges(
    ss,
    targetState,
    system: str = "simple_car",
    tolerance: float = 1e-6,
    nearest_match_max_dist: float | None = DEFAULT_NEAREST_VERTEX_MAX_DIST,
) -> tuple[list[ControlEdge], dict]:
    """Return duration-aware outgoing edges for a state in an OMPL tree.

    Malformed control edges are skipped and reported in metadata.  No invented
    fallback control is allowed because a fallback without the source edge's
    duration can violate both dynamics and collision constraints.
    """
    planner_data = oc.PlannerData(ss.getSpaceInformation())
    ss.getPlanner().getPlannerData(planner_data)
    num_vertices = int(planner_data.numVertices())
    metadata = {
        "match": "none",
        "vertex_index": None,
        "nearest_distance": float("inf"),
        "num_vertices": num_vertices,
        "num_children": 0,
        "skipped_edges": [],
    }
    if num_vertices == 0:
        log("[WARNING] Planner tree is empty", "warning")
        return [], metadata

    target = np.asarray(targetState, dtype=float).reshape(-1)
    target_vertex_index = None
    vertex_states: list[np.ndarray] = []
    distances: list[float] = []
    for vertex_index in range(num_vertices):
        vertex_state = np.asarray(
            state2list(planner_data.getVertex(vertex_index).getState(), system),
            dtype=float,
        )
        vertex_states.append(vertex_state)
        if isStateEqual(vertex_state, target, system, tolerance):
            target_vertex_index = vertex_index
            metadata.update(
                {
                    "match": "exact",
                    "vertex_index": vertex_index,
                    "nearest_distance": 0.0,
                }
            )
            break
        distances.append(float(arrayDistance(target, vertex_state, system)))

    if target_vertex_index is None:
        # The loop may have stopped early only on an exact match, so fill any
        # unvisited vertices before finding the nearest one.
        for vertex_index in range(len(vertex_states), num_vertices):
            vertex_state = np.asarray(
                state2list(planner_data.getVertex(vertex_index).getState(), system),
                dtype=float,
            )
            vertex_states.append(vertex_state)
            distances.append(float(arrayDistance(target, vertex_state, system)))
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        metadata.update(
            {
                "vertex_index": nearest_index,
                "nearest_distance": nearest_distance,
            }
        )
        if (
            nearest_match_max_dist is None
            or nearest_distance > float(nearest_match_max_dist)
        ):
            log(
                f"[WARNING] State {target.tolist()} not found in planner tree; "
                f"nearest distance is {nearest_distance:.6f}",
                "warning",
            )
            return [], metadata
        target_vertex_index = nearest_index
        metadata["match"] = "nearest"

    child_vertex_indices = getOutgoingEdgeIndices(
        planner_data, target_vertex_index
    )
    control_dimension = int(ss.getControlSpace().getDimension())
    si = ss.getSpaceInformation()
    step_size = float(si.getPropagationStepSize())
    minimum_steps = int(si.getMinControlDuration())
    maximum_steps = int(si.getMaxControlDuration())
    source_state = np.asarray(
        state2list(planner_data.getVertex(target_vertex_index).getState(), system),
        dtype=float,
    )

    children: list[ControlEdge] = []
    for child_vertex_index_raw in child_vertex_indices:
        child_vertex_index = int(child_vertex_index_raw)
        try:
            edge = planner_data.getEdge(target_vertex_index, child_vertex_index)
            control_obj = edge.getControl()
            control = np.asarray(
                [float(control_obj[index]) for index in range(control_dimension)],
                dtype=float,
            )
            duration_seconds = float(edge.getDuration())
            duration_steps = duration_seconds_to_steps(
                duration_seconds,
                step_size,
                min_steps=minimum_steps,
                max_steps=maximum_steps,
            )
            target_state = np.asarray(
                state2list(planner_data.getVertex(child_vertex_index).getState(), system),
                dtype=float,
            )
            children.append(
                ControlEdge(
                    source_state=source_state,
                    target_state=target_state,
                    control=control,
                    duration_steps=duration_steps,
                    duration_seconds=duration_seconds,
                    source_vertex=target_vertex_index,
                    target_vertex=child_vertex_index,
                    edge_id=f"{target_vertex_index}->{child_vertex_index}",
                )
            )
        except Exception as exc:
            metadata["skipped_edges"].append(
                {
                    "source_vertex": int(target_vertex_index),
                    "target_vertex": child_vertex_index,
                    "error": repr(exc),
                }
            )
            log(
                "[WARNING] Skipping malformed planner edge "
                f"{target_vertex_index}->{child_vertex_index}: {exc}",
                "warning",
            )

    metadata["num_children"] = len(children)
    return children, metadata


def getChildrenStates(
    ss,
    targetState,
    system="simple_car",
    tolerance=1e-6,
    nearest_match_max_dist: float | None = DEFAULT_NEAREST_VERTEX_MAX_DIST,
    return_metadata: bool = False,
):
    """Compatibility wrapper returning arrays while exposing typed edges in metadata."""
    edges, metadata = getChildEdges(
        ss,
        targetState,
        system=system,
        tolerance=tolerance,
        nearest_match_max_dist=nearest_match_max_dist,
    )
    states = [edge.target_state.copy() for edge in edges]
    controls = [edge.control.copy() for edge in edges]
    metadata["edges"] = edges
    if return_metadata:
        return states, controls, metadata
    return states, controls


def sampleRandomState(system, state, numStates=1000, posSTD=0.003, rotSTD=0.05):
    """Legacy state sampler retained for callers outside the core optimizer."""
    sampled_states = []
    if system in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
        if hasattr(state, "getX"):
            state_list = state2list(state, "SE2")
        else:
            state_list = np.asarray(state, dtype=float).reshape(-1)
        for _ in range(int(numStates)):
            sampled_states.append(
                [
                    float(state_list[0]) + np.random.normal(0.0, posSTD),
                    float(state_list[1]) + np.random.normal(0.0, posSTD),
                    (float(state_list[2]) + np.random.normal(0.0, rotSTD) + np.pi)
                    % (2.0 * np.pi)
                    - np.pi,
                ]
            )
        return sampled_states

    if system == "double_integrator":
        state_list = np.asarray(state, dtype=float).reshape(-1)
        if state_list.size < 6:
            raise ValueError("double_integrator state must have six values")
        for _ in range(int(numStates)):
            noisy = state_list[:6].copy()
            noisy[:3] += np.random.normal(0.0, posSTD, size=3)
            noisy[3:6] += np.random.normal(0.0, posSTD, size=3)
            sampled_states.append(noisy.tolist())
        return sampled_states

    if system in ("dublin_airplane", "dubins_airplane", "airplane"):
        state_list = np.asarray(
            state2list(state, "dubins_airplane")
            if not isinstance(state, (list, tuple, np.ndarray))
            else state,
            dtype=float,
        )
        for _ in range(int(numStates)):
            noisy = state_list.copy()
            noisy[:3] += np.random.normal(0.0, posSTD, size=3)
            noisy[3:5] += np.random.normal(0.0, rotSTD, size=2)
            noisy[3] = (noisy[3] + np.pi) % (2.0 * np.pi) - np.pi
            sampled_states.append(noisy.tolist())
        return sampled_states

    raise ValueError(f"Unsupported system for local sampling: {system}")
