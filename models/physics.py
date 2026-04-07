import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def push_physics(param, push_duration=3, obj_size=None):
    """Calculate the final state of the push given the param."""
    if isinstance(param, np.ndarray):
        param = torch.from_numpy(param)

    # Push parameters, Shape: (N, 1)
    rot, side, distance = param[:, 0:1], param[:, 1:2], param[:, 2:3]

    # Get the velocity at time t, Shape: (K, )
    k_steps = 100
    t = torch.linspace(0, push_duration, k_steps).to(param.device)
    vs, accs = sin_velocity_profile(t, distance, push_duration)

    # Calculate the final state
    x_final, y_final, theta_final = final_state(rot, side, vs, accs, t[1] - t[0], obj_size=obj_size)

    # Output dim (N, 3)
    return torch.stack([x_final, y_final, theta_final], dim=1)


def sin_velocity_profile(t, d, duration):
    """Get the velocity at time t for a given push distance d and duration."""
    t = t[None, :]  # Shape: (1, K)
    v_max = 2 * d / duration  # Shape: (N, 1)

    # Compute the velocity at time t for each sample given a sin velocity
    # Results have shape (N, K)
    v = (v_max / 2) * (torch.sin(2 * np.pi * t / duration - np.pi / 2) + 1)
    a = (v_max / 2) * torch.cos(2 * np.pi * t / duration - np.pi / 2) * (2 * np.pi / duration)

    return v, a


def progress_states(rot, side, velocities, accs, dt, obj_size=None):
    # Model from
    # Manipulation And Active Sensing By Pushing Using Tactile Feedback
    # https://ieeexplore.ieee.org/document/587370
    # Object properties, we assume they are unknown
    if obj_size is None:
        obj_size = 0.1  # Default value
    c = 0.05

    x_c = obj_size / 2
    y_c = side
    denom = c**2 + obj_size**2 + side**2

    # Pusher velocity, assume always perpendicular to the object
    # non-slip
    vpx = -velocities  # (N, K)
    vpy = 0.0

    # Compute constant body-frame object velocity
    vx = ((c**2 + x_c**2) * vpx + x_c * y_c * vpy) / denom
    vy = ((c**2 + y_c**2) * vpy + x_c * y_c * vpx) / denom
    omega = (x_c * vy - y_c * vx) / (c**2)

    # delta position and rotation in each step
    dx_step = vx * dt
    dy_step = vy * dt
    dtheta = omega * dt

    # Compute cumulative theta
    theta = torch.cumsum(dtheta, axis=1)
    # Compute cumulative object-frame displacement
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    dx_local = cos_theta * dx_step - sin_theta * dy_step
    dy_local = sin_theta * dx_step + cos_theta * dy_step

    # Cumulative local pose
    x_local = torch.cumsum(dx_local, axis=1)
    y_local = torch.cumsum(dy_local, axis=1)
    theta = torch.cumsum(dtheta, axis=1)

    # Position (rotate back to global given the initial rotation)
    cos_rot = torch.cos(rot)
    sin_rot = torch.sin(rot)
    x = (x_local * cos_rot) - (y_local * sin_rot)  # (N, K)
    y = (x_local * sin_rot) + (y_local * cos_rot)  # (N, K)

    return x, y, theta


def final_state(rot, side, velocities, accs, dt, obj_size=None):
    """Calculate the final state."""
    x, y, theta = progress_states(rot, side, velocities, accs, dt, obj_size=obj_size)
    x_final = x[:, -1]  # (N,)
    y_final = y[:, -1]  # (N,)
    theta_final = theta[:, -1]  # (N,)

    return x_final, y_final, theta_final


def visualize_process(param, x, y, theta, obj_size=0.1):
    rot, side, distance = (
        param[0, 0].item(),
        param[0, 1].item(),
        param[0, 2].item(),
    )
    x_np = x[0].detach().cpu().numpy()
    y_np = y[0].detach().cpu().numpy()
    theta_np = theta[0].detach().cpu().numpy()

    def get_square_corners(x_center, y_center, phi, size):
        half = size / 2.0
        local_corners = np.array(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ]
        )
        # Rotation matrix for angle phi.
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        global_corners = (local_corners @ R.T) + np.array([x_center, y_center])
        return global_corners

    # Set up the plot.
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.grid(True)

    # Plot the path of the object's center.
    ax.plot(x_np, y_np, "k--", label="Center Path")
    # Draw all the square states along the push.
    # Use a slight transparency so overlapping squares can be seen.
    for i in range(len(x_np)):
        # Compute the global orientation for this time step.
        phi = rot + theta_np[i]
        corners = get_square_corners(x_np[i], y_np[i], phi, obj_size)
        square = patches.Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor="blue",
            linewidth=1,
            alpha=0.5,
        )
        ax.add_patch(square)

    # Draw the push arrow.
    # We assume that the pusher contacts the object at a point defined in the
    contact_local = np.array([obj_size / 2, side])
    # Rotate the local contact point by the push’s global rotation (rot)
    R_init = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    contact_global = R_init @ contact_local
    # We assume the push is applied along the object's local negative x-axis.
    arrow_local = np.array([-distance, 0])
    arrow_global = R_init @ arrow_local
    ax.arrow(
        contact_global[0],
        contact_global[1],
        arrow_global[0],
        arrow_global[1],
        head_width=0.02,
        head_length=0.03,
        fc="red",
        ec="red",
        length_includes_head=True,
        label="Push Force",
    )

    # Set plot limits with a margin.
    margin = 0.15
    all_x = np.append(x_np, contact_global[0] + arrow_global[0])
    all_y = np.append(y_np, contact_global[1] + arrow_global[1])
    ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    ax.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
    ax.legend()
    ax.set_title("Static Visualization of a Pushed Square Object")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    # Set up an example push
    push_param = [0, 0.02, 0.3]
    push_duration = 3
    param = torch.tensor([push_param])
    rot, side, distance = param[:, 0:1], param[:, 1:2], param[:, 2:3]

    k_steps = 100
    t = torch.linspace(0, push_duration, k_steps)
    vs, accs = sin_velocity_profile(t, distance, push_duration)

    # Get states
    x, y, theta = progress_states(rot, side, vs, accs, t[1] - t[0], obj_size=0.1)

    # Visualize it
    visualize_process(param, x, y, theta, obj_size=0.1)
