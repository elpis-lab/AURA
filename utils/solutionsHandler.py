from ompl import base as ob
from ompl import control as oc

from utils.utils import printState


def getPathInfo(solution_path, ss):
    solution_info = {}
    solution_info["state_count"] = solution_path.getStateCount()
    solution_info["control_count"] = solution_path.getControlCount()

    control_space = ss.getControlSpace()
    control_dimension = control_space.getDimension()

    # Extract all controls while objects are valid
    controls_list = []
    # Extract all control durations
    time_list = []
    for i in range(solution_info["control_count"]):
        control = solution_path.getControl(i)
        control_values = [control[j] for j in range(control_dimension)]
        controls_list.append(control_values)

        # Get the duration for this control
        control_duration = solution_path.getControlDuration(i)
        time_list.append(control_duration)

    solution_info["controls"] = controls_list
    solution_info["time"] = time_list

    # Extract all states while objects are valid
    states_list = []
    space_type = ss.getSpaceInformation().getStateSpace().getType()

    for i in range(solution_info["state_count"]):
        state = solution_path.getState(i)

        if space_type == ob.STATE_SPACE_SE2:
            state_values = [
                state.getX(),
                state.getY(),
                state.getYaw(),
            ]
        elif space_type == ob.STATE_SPACE_SE3:
            state_values = [
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
        else:
            # Handle RealVector (e.g., double_integrator) and compound spaces.
            dim = ss.getSpaceInformation().getStateSpace().getDimension()
            state_values = []

            # RealVectorStateSpace is directly indexable as state[j].
            # Handle this first to avoid incorrect compound extraction attempts.
            try:
                if space_type == ob.STATE_SPACE_REAL_VECTOR:
                    state_values = [float(state[j]) for j in range(dim)]
                else:
                    raise ValueError("not real-vector")
            except Exception:
                # Try compound extraction (e.g., dublin_airplane: R^3 x SO(3))
                try:
                    state_values = [
                        state[0][0],  # x from R^3 component
                        state[0][1],  # y from R^3 component
                        state[0][2],  # z from R^3 component
                        state[1].w,  # qw from SO(3) component
                        state[1].x,  # qx from SO(3) component
                        state[1].y,  # qy from SO(3) component
                        state[1].z,  # qz from SO(3) component
                    ]
                except Exception:
                    # Final generic extraction fallback.
                    try:
                        state_values = [float(state[j]) for j in range(dim)]
                    except Exception:
                        state_values = [0.0] * dim

        states_list.append(state_values)
    solution_info["states"] = states_list

    return solution_info


def getAllPlannerSolutionsInfo(planner, ss):
    """Extract detailed information from all solutions stored in the planner."""
    try:
        all_solutions = planner.getAllSolutions()
        all_solution_infos = []

        for i, solution in enumerate(all_solutions):
            try:
                # Create PathControl from the solution path
                path_control = oc.PathControl(solution.path_)

                # Extract path information
                info = getPathInfo(path_control, ss)
                info["cost"] = solution.cost_.value()
                info["solution_index"] = i

                all_solution_infos.append(info)

            except Exception as e:
                print(f"❌ Error processing solution {i+1}: {e}")
                # Create a minimal info structure
                info = {
                    "state_count": 0,
                    "control_count": 0,
                    "states": [],
                    "controls": [],
                    "cost": solution.cost_.value(),
                    "solution_index": i,
                    "error": str(e),
                }
                all_solution_infos.append(info)

        # Sort by cost (best first)
        all_solution_infos.sort(key=lambda x: x["cost"])

        return all_solution_infos

    except Exception as e:
        print(f"❌ Error getting solutions from planner: {e}")
        return []


def printAllPlannerSolutions(planner, title="All Planner Solutions"):
    """Print a summary of all solutions tracked by the planner."""
    print(f"\n📊 {title}")
    print("=" * 60)
    try:
        all_solutions = planner.getAllSolutions()
        print(f"Number of solutions found: {len(all_solutions)}")
        for i, solution in enumerate(all_solutions):
            print(f"  Solution {i+1}:")
            print(f"    Cost: {solution.cost_.value():.3f}")
            try:
                path_control = oc.PathControl(solution.path_)
                if path_control:
                    print(f"    States: {path_control.getStateCount()}")
                    print(f"    Controls: {path_control.getControlCount()}")
                else:
                    print(f"    Path: Unable to cast to PathControl")
            except Exception as e:
                print(f"    Path: Error accessing path ({e})")
    except Exception as e:
        print(f"❌ Error getting solutions from planner: {e}")
    print("=" * 60)


def getBestSolutionFromPlanner(ss):
    """Get the best solution using the new planner tracking feature, with fallback."""
    planner = ss.getPlanner()

    # Try to use the new solution tracking feature first
    if hasattr(planner, "getAllSolutions"):
        try:
            all_solution_infos = getAllPlannerSolutionsInfo(planner, ss)
            if len(all_solution_infos) > 0:
                # Return the best solution (first one after sorting by cost)
                return all_solution_infos[0], all_solution_infos
            else:
                print(
                    "⚠️ Enhanced tracking returned no solutions, falling back to original method..."
                )
        except Exception as e:
            print(
                f"⚠️ Error with enhanced solution tracking: {e}, falling back to original method..."
            )

    # Fallback to original method
    infos = getSolutionsInfo(ss)
    if len(infos) > 0:
        return infos[0], infos
    else:
        return None, []


def printBestSolution(best_solution, iteration_label=""):
    """Print detailed information about the best solution."""
    if not best_solution:
        print(f"❌ {iteration_label}: No solution available")
        return

    print(f"\n🏆 {iteration_label} - BEST SOLUTION:")
    print("=" * 50)
    print(f"  Cost: {best_solution['cost']:.5f}")
    print(f"  States: {best_solution['state_count']}")
    print(f"  Controls: {best_solution['control_count']}")

    print(f"\n  📍 States:")
    for i, state in enumerate(best_solution["states"]):
        printState(state, "SE2", f"    [{i}]")

    print(f"\n  🎮 Controls:")
    if "controls" in best_solution and len(best_solution["controls"]) > 0:
        for i, control in enumerate(best_solution["controls"]):
            if "time" in best_solution and i < len(best_solution["time"]):
                duration = best_solution["time"][i]
                printState(control, "RealVector", f"    [{i}]")
                print(f"    (duration: {duration:.3f}s)")
            else:
                printState(control, "RealVector", f"    [{i}]")
    else:
        print("    (No controls - reached goal state)")
    print("=" * 50)


def getSolutionsInfo(ss):
    """
    Enhanced function to get solution information using the new planner solution tracking feature.
    Falls back to the original method if the new feature is not available.
    """
    planner = ss.getPlanner()

    # Try to use the new solution tracking feature first
    try:
        if hasattr(planner, "getAllSolutions"):
            all_solution_infos = getAllPlannerSolutionsInfo(planner, ss)
            if len(all_solution_infos) > 0:
                return all_solution_infos
    except Exception as e:
        print(f"⚠️ Error with enhanced solution tracking: {e}, falling back to original method...")

    # Fallback to original method
    solutions = ss.getProblemDefinition().getSolutions()
    allSolutionInfos = []

    # First try to get solutions from problem definition
    for solution in solutions:
        info = getPathInfo(solution.path_, ss)
        info["cost"] = solution.cost_.value()
        allSolutionInfos.append(info)

    # If no solutions found in problem definition, try to get the solution path directly
    if len(allSolutionInfos) == 0:
        try:
            solution_path = ss.getSolutionPath()
            if solution_path and solution_path.getStateCount() > 0:
                info = getPathInfo(solution_path, ss)
                # Try to get cost from optimization objective
                try:
                    opt = ss.getProblemDefinition().getOptimizationObjective()
                    if opt:
                        info["cost"] = opt.cost(solution_path).value()
                    else:
                        info["cost"] = 0.0  # Default cost for approximate solutions
                except:
                    info["cost"] = 0.0  # Default cost for approximate solutions
                allSolutionInfos.append(info)
        except Exception as e:
            print(f"❌ Error getting solution path directly: {e}")

    # Sort by cost (best first)
    allSolutionInfos.sort(key=lambda x: x["cost"])

    return allSolutionInfos
