# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John-Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def animate_formation(
    t, chief_state, deputies_states, save_path="formation_animation.gif", fps=30
):
    """
    Generates and saves a 3D animation of the formation flight.

    Args:
        t (array): Time vector.
        chief_state (array): (N, 6) array of chief state [x, y, z, vx, vy, vz].
        deputies_states (list of arrays): List of (N, 6) arrays for each deputy.
        save_path (str): Path to save the animation.
        fps (int): Frames per second.
    """
    print("Generating Formation Animation...")

    cx, cy, cz = chief_state[:, 0], chief_state[:, 1], chief_state[:, 2]

    followers = []
    for dep in deputies_states:
        followers.append((dep[:, 0], dep[:, 1], dep[:, 2]))

    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Limits
    all_x = np.concatenate([cx] + [f[0] for f in followers])
    all_y = np.concatenate([cy] + [f[1] for f in followers])
    all_z = np.concatenate([cz] + [f[2] for f in followers])

    ax.set_xlim(np.min(all_x), np.max(all_x))
    ax.set_ylim(np.min(all_y), np.max(all_y))
    ax.set_zlim(np.min(all_z), np.max(all_z))

    ax.set_xlabel("X (Rotating)")
    ax.set_ylabel("Y (Rotating)")
    ax.set_zlabel("Z (Rotating)")
    ax.set_title("Formation Flight Animation")

    # Lines and Dots
    # Chief
    (line_c,) = ax.plot([], [], [], "k-", linewidth=1, alpha=0.5)
    (dot_c,) = ax.plot([], [], [], "ko", label="Chief")

    # Followers
    lines_f = []
    dots_f = []
    colors = ["r", "g", "b", "m", "c", "y"]

    for i in range(len(followers)):
        color = colors[i % len(colors)]
        (l,) = ax.plot(
            [], [], [], color=color, linestyle="--", linewidth=0.5, alpha=0.5
        )
        (d,) = ax.plot([], [], [], color=color, marker="o", label=f"Deputy {i+1}")
        lines_f.append(l)
        dots_f.append(d)

    ax.legend()

    # Animation Function
    # Downsample for speed
    skip = max(1, len(t) // (10 * fps))  # Approx 10 seconds duration
    num_frames = len(t) // skip

    def update(frame):
        idx = frame * skip

        # Update Chief
        line_c.set_data(cx[:idx], cy[:idx])
        line_c.set_3d_properties(cz[:idx])
        dot_c.set_data([cx[idx]], [cy[idx]])
        dot_c.set_3d_properties([cz[idx]])

        # Update Followers
        for i in range(len(followers)):
            fx, fy, fz = followers[i]
            lines_f[i].set_data(fx[:idx], fy[:idx])
            lines_f[i].set_3d_properties(fz[:idx])
            dots_f[i].set_data([fx[idx]], [fy[idx]])
            dots_f[i].set_3d_properties([fz[idx]])

        return line_c, dot_c, *lines_f, *dots_f

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000 / fps, blit=False
    )

    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer="pillow", fps=fps)
    print("Animation saved!")
