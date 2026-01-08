import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def obs_to_theta(obs):
    """
    obs: (T, 3) or (..., 3) = [cosθ, sinθ, θ_dot]
    return: (T, 2) = [θ, θ_dot]
    """
    cos_t = obs[..., 0]
    sin_t = obs[..., 1]
    theta = np.arctan2(sin_t, cos_t)
    theta_dot = obs[..., 2]
    return np.stack([theta, theta_dot], axis=-1)


def plot_pendulum_trajectories(trajs, save_path=None):
    """
    trajs: list of np.array, each (T, 3)
    """
    plt.figure(figsize=(6, 6))

    for traj in trajs:
        th = obs_to_theta(traj)
        plt.plot(th[:, 0], th[:, 1], lw=2, alpha=0.8)

    plt.scatter(0, 0, s=300, marker="*", c="gold", edgecolors="black", label="Goal")
    plt.xlabel("θ")
    plt.ylabel("θ̇")
    plt.title("Pendulum trajectories (θ–θ̇)")
    plt.legend()
    plt.axis("equal")

    if save_path:
        plt.savefig(os.path.join(save_path, "traj_theta_phase.png"))
        plt.close()
    else:
        plt.show()


def plot_q_gradient_field_pendulum(
    actor,
    q_net,
    device,
    trajs=None,
    theta_lim=np.pi,
    theta_dot_lim=8.0,
    grid_size=25,
    save_path=None,
):
    actor.eval()
    q_net.eval()

    thetas = torch.linspace(-theta_lim, theta_lim, grid_size, device=device)
    theta_dots = torch.linspace(-theta_dot_lim, theta_dot_lim, grid_size, device=device)

    grid = []
    grads = []

    for th in thetas:
        for thd in theta_dots:
            obs = torch.tensor(
                [[torch.cos(th), torch.sin(th), thd]],
                device=device,
                requires_grad=True,
            )

            with torch.enable_grad():
                a, _ = actor.sample(obs)
                q = q_net(obs, a)
                g = torch.autograd.grad(q, obs)[0][0]

            # chain rule: dQ/dθ
            dQ_dtheta = -torch.sin(th) * g[0] + torch.cos(th) * g[1]
            dQ_dtheta_dot = g[2]

            grid.append([th.item(), thd.item()])
            grads.append([dQ_dtheta.item(), dQ_dtheta_dot.item()])

    grid = np.array(grid)
    grads = np.array(grads)

    plt.figure(figsize=(6, 6))
    plt.quiver(
        grid[:, 0],
        grid[:, 1],
        grads[:, 0],
        grads[:, 1],
        angles="xy",
        scale_units="xy",
        scale=20,
        alpha=0.9,
    )

    # goal
    plt.scatter(0, 0, s=300, marker="*", c="gold", edgecolors="black")

    # overlay trajectories
    if trajs is not None:
        for traj in trajs:
            th = obs_to_theta(traj)
            plt.plot(th[:, 0], th[:, 1], lw=2, alpha=0.8)

    plt.xlabel("θ")
    plt.ylabel("θ̇")
    plt.title(r"$\nabla_{(\theta,\dot\theta)} Q(s,\pi(s))$")
    plt.axis("equal")

    if save_path:
        plt.savefig(os.path.join(save_path, "q_gradient_field_pendulum.png"))
        plt.close()
    else:
        plt.show()


def pendulum_distance(theta_state, alpha=0.1):
    """
    theta_state: (T, 2) = [θ, θ_dot]
    """
    return np.sqrt(theta_state[:, 0] ** 2 + alpha * theta_state[:, 1] ** 2)


def visualize_pendulum_results(
    actor,
    q_net,
    device,
    trajs,
    save_path=None,
):
    plot_pendulum_trajectories(trajs, save_path)
    plot_q_gradient_field_pendulum(
        actor,
        q_net,
        device,
        trajs=trajs,
        save_path=save_path,
    )
