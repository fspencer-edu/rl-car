import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from rich.console import Console
from rich.table import Table

from agent import moving_average

console = Console()


def plot_training(rewards, steps_history, success_history):
    plt.figure(figsize=(9, 4))
    plt.plot(rewards, alpha=0.35, label="episode reward")
    ma = moving_average(rewards, 25)
    if len(ma) > 0:
        plt.plot(range(len(ma)), ma, label="moving avg")
    plt.title("Car RL Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(steps_history, alpha=0.45, label="steps")
    ma = moving_average(steps_history, 25)
    if len(ma) > 0:
        plt.plot(range(len(ma)), ma, label="moving avg")
    plt.title("Steps Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    sr = moving_average(success_history, 25)
    plt.plot(range(len(sr)), sr)
    plt.title("Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def draw_world(ax, env):
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#f4f6f8")
    ax.grid(True, alpha=0.2)

    for ox, oy, ow, oh in env.obstacles:
        ax.add_patch(Rectangle((ox, oy), ow, oh, alpha=0.7))

    goal = Circle(env.goal, env.goal_radius, alpha=0.5)
    ax.add_patch(goal)
    ax.text(env.goal[0], env.goal[1], "GOAL", ha="center", va="center", fontsize=10, weight="bold")

    start = Circle(env.start, 0.25, alpha=0.5)
    ax.add_patch(start)
    ax.text(env.start[0], env.start[1] - 0.45, "START", ha="center", va="center", fontsize=9)


def animate_learning(env, snapshot_paths, snapshot_meta):
    selected_eps = sorted(snapshot_paths.keys())
    frames = []
    titles = []

    for ep in selected_eps:
        path = snapshot_paths[ep]
        info = snapshot_meta[ep]
        heading = 0.0

        for step_idx, pos in enumerate(path):
            if step_idx > 0:
                delta = path[step_idx] - path[step_idx - 1]
                if np.linalg.norm(delta) > 1e-8:
                    heading = math.atan2(delta[1], delta[0])

            frames.append((ep, step_idx, pos, heading, path, info))
            titles.append(
                f"Episode {ep + 1} demo | step {step_idx + 1}/{len(path)} | "
                f"success={info['success']} | reward={info['reward']:.2f}"
            )

        for _ in range(8):
            frames.append((ep, len(path) - 1, path[-1], heading, path, info))
            titles.append(
                f"Episode {ep + 1} demo | final | "
                f"success={info['success']} | reward={info['reward']:.2f}"
            )

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_world(ax, env)

    trail_line, = ax.plot([], [], linewidth=2, alpha=0.7)

    car_body = Rectangle((0, 0), 0.5, 0.28, angle=0.0, rotation_point="center")
    ax.add_patch(car_body)

    car_arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=14, linewidth=2)
    ax.add_patch(car_arrow)

    title = ax.set_title("Car learning animation")

    def init():
        trail_line.set_data([], [])
        car_body.set_xy((env.start[0] - 0.25, env.start[1] - 0.14))
        title.set_text("Car learning animation")
        return trail_line, car_body, car_arrow, title

    def update(frame_idx):
        ep, step_idx, pos, heading, path, info = frames[frame_idx]

        xs = [p[0] for p in path[:step_idx + 1]]
        ys = [p[1] for p in path[:step_idx + 1]]
        trail_line.set_data(xs, ys)

        car_body.set_xy((pos[0] - 0.25, pos[1] - 0.14))
        car_body.angle = np.degrees(heading)

        arrow_len = 0.55
        end_x = pos[0] + math.cos(heading) * arrow_len
        end_y = pos[1] + math.sin(heading) * arrow_len
        car_arrow.set_positions((pos[0], pos[1]), (end_x, end_y))

        title.set_text(titles[frame_idx])
        return trail_line, car_body, car_arrow, title

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=140,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()
    return anim


def print_summary(rewards, steps_history, success_history):
    table = Table(title="Car RL Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Episodes", str(len(rewards)))
    table.add_row("Final reward", f"{rewards[-1]:.2f}")
    table.add_row("Best reward", f"{np.max(rewards):.2f}")
    table.add_row("Final steps", str(steps_history[-1]))
    recent = np.mean(success_history[-50:]) if len(success_history) >= 50 else np.mean(success_history)
    table.add_row("Recent success rate", f"{recent:.2f}")
    console.print(table)
