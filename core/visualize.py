import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch
import sac
import numpy as np
import scipy.fft
import matplotlib.cm as cm
import torch.nn as nn

def plot_qf(actor, q1, device, trajs=None, save_path=None):
    xs = torch.linspace(-4, 4, 25).to(device)
    ys = torch.linspace(-4, 4, 25).to(device)

    grid, grad = sac.q_state_gradient_field(actor, q1, xs, ys)

    grid = grid.cpu().numpy()
    grad = grad.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.quiver(
        grid[:, 0], grid[:, 1],
        grad[:, 0], grad[:, 1],
        angles="xy"
    )
    plt.scatter(0, 0, c="yellow", s=300, marker='*', edgecolors='black', linewidths=2, label="Goal")

    if trajs is not None:
        if isinstance(trajs, np.ndarray):
            trajs = [trajs]
        colors = cm.rainbow(np.linspace(0, 1, 10))
        for i, traj in enumerate(trajs):
            if i >= 10:
                break
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.8, label=f'Ep {i}')
            plt.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=60, edgecolors='black', zorder=5)
            plt.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker='X', s=60, edgecolors='black', zorder=5)
    
    plt.legend()
    plt.axis("equal")
    plt.title("∇s Q(s, π(s)) vector field")
    if save_path:
            plt.savefig(os.path.join(save_path, "q_field.png"))
            plt.close()
    else:
        plt.show()

import numpy as np


def smoothness_score(actions, dt=0.1, print_score=True):
    if isinstance(actions, np.ndarray):
        actions = [actions]
        
    fs = 1.0 / dt
    scores = []

    for action in actions:
        a = np.array(action, dtype=float)
        if a.ndim == 1:
            a = a[:, None]

        n, d = a.shape
        if n < 2:
            scores.append(0.0)
            continue

        yf = np.fft.fft(a, axis=0)
        cutoff = n // 2
        yf = yf[:cutoff, :]
        
        freqs = np.fft.fftfreq(n, d=dt)[:cutoff]
        freqs = freqs.reshape(-1, 1)
        
        weighted_sum = np.sum(freqs * np.abs(yf), axis=0) 
        smooth_per_dim = (2.0 / (n * fs)) * weighted_sum
        scores.append(float(np.mean(smooth_per_dim)))

    mean_score = np.mean(scores) if scores else 0.0
    if print_score:
        print("Mean smoothness score:", mean_score)
    
    return mean_score, scores


def plot_grad_a(actor, q1, device, trajs=None, step_freq=10, save_path=None):
    xs = torch.linspace(-4, 4, 25).to(device)
    ys = torch.linspace(-4, 4, 25).to(device)

    grid, grad = sac.q_state_gradient_field(actor, q1, xs, ys)

    grid = grid.cpu().numpy()
    grad = grad.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.quiver(
        grid[:, 0], grid[:, 1],
        grad[:, 0], grad[:, 1],
        angles="xy", color='gray', alpha=0.2,
    )
    plt.scatter(0, 0, c="yellow", s=300, marker='*', edgecolors='black', linewidths=2, label="Goal")

    if trajs is not None:
        if isinstance(trajs, np.ndarray):
            trajs = [trajs]
        colors = cm.rainbow(np.linspace(0, 1, 10))
        _, grad_a_list = sac.q_action_gradient_field(actor, q1, trajs, device)

        alignment_score_list = []
        for i, (traj, grad_a) in enumerate(zip(trajs, grad_a_list)):
            if i >= 10:
                break
            score = alignment_score(traj, grad_a)
            print(f"[Ep {i}] Alignment score (∇a Q): {score:.4f}")
            alignment_score_list.append(score)
        print(f"[Test] Mean Alignment score {np.nanmean(alignment_score_list):.2f}")

        for i, (traj, grad_a) in enumerate(zip(trajs, grad_a_list)):
            if i >= 10:
                break
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.8, label=f'Ep {i}')
            plt.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=60, edgecolors='black', zorder=5)
            plt.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker='X', s=60, edgecolors='black', zorder=5)

            idx = np.arange(0, len(traj), step_freq)
            q_x = traj[idx, 0]
            q_y = traj[idx, 1]
            q_u = grad_a[idx, 0].cpu().numpy()
            q_v = grad_a[idx, 1].cpu().numpy()
            plt.quiver(
                q_x, q_y,
                q_u, q_v,
                angles="xy", color=colors[i],
                edgecolors='black', linewidths=1,
            )
    
    plt.legend()
    plt.axis("equal")
    plt.title("∇a Q(s, π(s)) vector field")
    if save_path:
            plt.savefig(os.path.join(save_path, "grad_a.png"))
            plt.close()
    else:
        plt.show()

    plot_alignment_colormap(
        trajs,
        grad_a_list,
        title="∇a Q alignment",
        save_path=os.path.join(save_path, "grad_a_align_cmap.png") if save_path else None
    )
    plot_alignment_vs_distance(
        trajs,
        grad_a_list,
        title="∇a Q alignment vs distance",
        save_path=os.path.join(save_path, "grad_a_align_") if save_path else None
    )


def plot_grad_a_target(actor, q1, q2, dynamics, device, trajs=None, step_freq=10, save_path=None):
    xs = torch.linspace(-4, 4, 25).to(device)
    ys = torch.linspace(-4, 4, 25).to(device)

    grid, grad = sac.q_state_gradient_field(actor, q1, xs, ys)

    grid = grid.cpu().numpy()
    grad = grad.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.quiver(
        grid[:, 0], grid[:, 1],
        grad[:, 0], grad[:, 1],
        angles="xy", color='gray', alpha=0.2,
    )
    plt.scatter(0, 0, c="yellow", s=300, marker='*', edgecolors='black', linewidths=2, label="Goal")

    if trajs is not None:
        if isinstance(trajs, np.ndarray):
            trajs = [trajs]
        colors = cm.rainbow(np.linspace(0, 1, 10))
        _, grad_a_list = sac.q_model_based_action_gradient_field(actor, q1, q2, dynamics, trajs, device)

        alignment_score_list = []
        for i, (traj, grad_a) in enumerate(zip(trajs, grad_a_list)):
            if i >= 10:
                break
            score = alignment_score(traj, grad_a)
            print(f"[Ep {i}] Alignment score (Target ∇a Q): {score:.4f}")
            alignment_score_list.append(score)
        print(f"[Test] Mean Alignment score {np.nanmean(alignment_score_list):.2f}")
            
        for i, (traj, grad_a) in enumerate(zip(trajs, grad_a_list)):
            if i >= 10:
                break
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.8, label=f'Ep {i}')
            plt.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=60, edgecolors='black', zorder=5)
            plt.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker='X', s=60, edgecolors='black', zorder=5)

            idx = np.arange(0, len(traj), step_freq)
            q_x = traj[idx, 0]
            q_y = traj[idx, 1]
            q_u = grad_a[idx, 0].cpu().numpy()
            q_v = grad_a[idx, 1].cpu().numpy()
            plt.quiver(
                q_x, q_y,
                q_u, q_v,
                angles="xy", color=colors[i],
                edgecolors='black', linewidths=1,
            )
    
    plt.legend()
    plt.axis("equal")
    plt.title("∇a Q(s, π(s)) vector field")
    if save_path:
            plt.savefig(os.path.join(save_path, "target_grad_a.png"))
            plt.close()
    else:
        plt.show()

    plot_alignment_colormap(
        trajs,
        grad_a_list,
        title="Target ∇a Q alignment",
        save_path=os.path.join(save_path, "target_grad_a_align_cmap.png") if save_path else None
    )

    plot_alignment_vs_distance(
        trajs,
        grad_a_list,
        title="Target ∇a Q alignment vs distance",
        save_path=os.path.join(save_path, "target_grad_a_align_") if save_path else None
    )


def alignment_score(points, grads, goal=np.array([0.0, 0.0]), eps=1e-8):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(grads, torch.Tensor):
        grads = grads.detach().cpu().numpy()
    
    goal_dir = goal[None, :] - points  # (N, 2)
    goal_norm = np.linalg.norm(goal_dir, axis=1, keepdims=True)
    grad_norm = np.linalg.norm(grads, axis=1, keepdims=True)

    # zero division
    valid = (goal_norm[:, 0] > eps) & (grad_norm[:, 0] > eps)
    if valid.sum() == 0:
        return np.nan

    goal_dir = goal_dir[valid]
    grads = grads[valid]

    # cosine similarity
    cos = np.sum(goal_dir * grads, axis=1) / (
        np.linalg.norm(goal_dir, axis=1) * np.linalg.norm(grads, axis=1)
    )

    return cos.mean()


def stepwise_alignment(points, grads, goal=np.array([0.0, 0.0]), eps=1e-8):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(grads, torch.Tensor):
        grads = grads.detach().cpu().numpy()

    goal_dir = goal[None, :] - points
    goal_norm = np.linalg.norm(goal_dir, axis=1)
    grad_norm = np.linalg.norm(grads, axis=1)

    valid = (goal_norm > eps) & (grad_norm > eps)
    if valid.sum() == 0:
        return None, None
    
    goal_dir = goal_dir[valid]
    goal_norm = goal_norm[valid]
    grads = grads[valid]

    # cosine similarity
    cos = np.sum(goal_dir * grads, axis=1) / (
        np.linalg.norm(goal_dir, axis=1) * np.linalg.norm(grads, axis=1)
    )

    return goal_norm, cos


def binned_mean(dist, values, bins=12):
    bins_edges = np.linspace(dist.min(), dist.max(), bins + 1)
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

    mean_vals = []
    for i in range(bins):
        mask = (dist >= bins_edges[i]) & (dist < bins_edges[i + 1])
        if mask.sum() == 0:
            mean_vals.append(np.nan)
        else:
            mean_vals.append(np.mean(values[mask]))

    return bin_centers, np.array(mean_vals)


def binned_alignment(dist, align, bins=10, use_median=True):
    bin_edges = np.linspace(dist.min(), dist.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    stats = []
    for i in range(bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        if mask.sum() == 0:
            stats.append(np.nan)
        else:
            vals = align[mask]
            stats.append(np.median(vals) if use_median else np.mean(vals))

    return bin_centers, np.array(stats)


def binned_variance(dist, align, bins=10):
    bin_edges = np.linspace(dist.min(), dist.max(), bins + 1)
    centers, vars = [], []

    for i in range(bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i+1])
        if mask.sum() > 1:
            centers.append(dist[mask].mean())
            vars.append(np.var(align[mask]))

    return np.array(centers), np.array(vars)


def plot_alignment_colormap(trajs, grad_a_list, title="Alignment colormap", save_path=None):
    all_xy = []
    all_align = []

    for traj, grad_a in zip(trajs, grad_a_list):
        dist, align = stepwise_alignment(traj, grad_a)
        if dist is None:
            continue

        all_xy.append(traj[: len(align)])
        all_align.append(align)

    if len(all_xy) == 0:
        print("No valid alignment data")
        return

    all_xy = np.concatenate(all_xy, axis=0)
    all_align = np.concatenate(all_align, axis=0)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        all_xy[:, 0],
        all_xy[:, 1],
        c=all_align,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=25,
        alpha=0.8,
    )

    plt.scatter(0, 0, c="yellow", s=300, marker="*", edgecolors="black", linewidths=2)
    plt.colorbar(sc, label="Alignment score (cos)")
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    if save_path:
            plt.savefig(save_path)
            plt.close()
    else:
        plt.show()


def plot_alignment_vs_distance(trajs, grad_a_list, title="Alignment vs Distance", save_path=None):
    all_dist = []
    all_align = []

    for traj, grad_a in zip(trajs, grad_a_list):
        dist, align = stepwise_alignment(traj, grad_a)
        if dist is None:
            continue

        all_dist.append(dist)
        all_align.append(align)

    if len(all_dist) == 0:
        print("No valid alignment data")
        return

    all_dist = np.concatenate(all_dist)
    all_align = np.concatenate(all_align)

    # --------alignment and linear fitting--------
    plt.figure(figsize=(6, 4))
    plt.scatter(all_dist, all_align, s=18, alpha=0.4)

    coef = np.polyfit(all_dist, all_align, 1)
    x = np.linspace(all_dist.min(), all_dist.max(), 200)
    y = coef[0] * x + coef[1]

    plt.plot(x, y, "r", lw=2, label=f"slope = {coef[0]:.3f}")
    plt.axhline(0, color="gray", ls="--", alpha=0.5)

    plt.xlabel("Distance to goal")
    plt.ylabel("Alignment score")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path:
            plt.savefig(f"{save_path}regress.png")
            plt.close()
    else:
        plt.show()

    # --------binned alignment median and mean--------
    x, y = binned_alignment(all_dist, all_align, bins=48)
    x_m, y_m = binned_mean(all_dist, all_align, bins=48)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, "o-", lw=2, label="binned median")
    plt.plot(x_m, y_m, "s--", lw=2, label="binned mean")
    plt.axhline(0, color="gray", ls="--", alpha=0.5)

    plt.xlabel("Distance to goal")
    plt.ylabel("Alignment score median, mean")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path:
            plt.savefig(f"{save_path}median_mean.png")
            plt.close()
    else:
        plt.show()

    # --------binned alignment variance--------
    x, y = binned_variance(all_dist, all_align, bins=48)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, "o-", lw=2)
    plt.axhline(0, color="gray", ls="--", alpha=0.5)

    plt.xlabel("Distance to goal")
    plt.ylabel("Alignment score variance")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path:
            plt.savefig(f"{save_path}variance.png")
            plt.close()
    else:
        plt.show()


def plot_smoothness_vs_distance(
    trajectories,      # list of (T, state_dim)
    actions_list,      # list of (T, action_dim)
    goal=np.array([0.0, 0.0]),
    window_size=32,
    stride=1,
    num_bins=48,
    save_path=None,
    save_data=True,
):
    """
    For each trajectory:
      - slide a fixed-size time window
      - compute action smoothness within each window
      - x-axis: mean distance to goal in the window
      - y-axis: smoothness score of actions in the window

    All (distance, smoothness) pairs across trajectories are aggregated.
    Distance-binned mean & median trends are overlaid, with a std band.
    """

    all_distances = []
    all_smoothness = []

    # --------------------------------------------------
    # Sliding-window smoothness computation
    # --------------------------------------------------
    for traj, actions in zip(trajectories, actions_list):
        T = min(len(traj), len(actions))
        traj = traj[:T]
        actions = actions[:T]

        distances = np.linalg.norm(traj - goal, axis=1)
        win = min(window_size, T)

        for start in range(0, T - win + 1, stride):
            end = start + win

            action_seq = actions[start:end]
            dist_seq = distances[start:end]

            sm, _ = smoothness_score(action_seq, print_score=False)
            if np.isnan(sm):
                continue

            all_distances.append(np.mean(dist_seq))
            all_smoothness.append(sm)

    all_distances = np.asarray(all_distances)
    all_smoothness = np.asarray(all_smoothness)

    if len(all_distances) == 0:
        print("No valid smoothness values computed.")
        return

    # --------------------------------------------------
    # Distance binning (based on valid samples)
    # --------------------------------------------------
    d_min = all_distances.min()
    d_max = all_distances.max()

    bin_edges = np.linspace(d_min, d_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_means = []
    bin_medians = []
    bin_stds = []

    for i in range(num_bins):
        mask = (all_distances >= bin_edges[i]) & (all_distances < bin_edges[i + 1])
        vals = all_smoothness[mask]

        if len(vals) == 0:
            bin_means.append(np.nan)
            bin_medians.append(np.nan)
            bin_stds.append(np.nan)
        else:
            bin_means.append(np.mean(vals))
            bin_medians.append(np.median(vals))
            bin_stds.append(np.std(vals))

    bin_means = np.asarray(bin_means)
    bin_medians = np.asarray(bin_medians)
    bin_stds = np.asarray(bin_stds)

    if save_path is not None and save_data:
        np.savez(
            os.path.join(save_path, "smoothness_vs_distance_bins.npz"),
            bin_centers=bin_centers,
            bin_means=bin_means,
            bin_medians=bin_medians,
            bin_stds=bin_stds,
            num_bins=num_bins,
            window_size=window_size,
            stride=stride,
        )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    # Raw scatter
    plt.scatter(
        all_distances,
        all_smoothness,
        s=12,
        alpha=0.35,
        edgecolors="none",
        label="Sliding-window samples",
    )

    # Mean trend
    plt.plot(
        bin_centers,
        bin_means,
        color="red",
        lw=2,
        label="Binned mean",
    )

    # Median trend
    plt.plot(
        bin_centers,
        bin_medians,
        color="black",
        lw=2,
        linestyle="--",
        label="Binned median",
    )

    # Std band (around mean)
    plt.fill_between(
        bin_centers,
        bin_means - bin_stds,
        bin_means + bin_stds,
        color="red",
        alpha=0.25,
        label="±1 std",
    )

    plt.xlabel("Mean distance to goal (window)")
    plt.ylabel("Action smoothness")
    plt.title(f"Sliding-window smoothness vs distance (window={window_size})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "smoothness_vs_distance.png"))
        plt.close()
    else:
        plt.show()


def plot_fisher_vs_distance(
    trajectories,  # list of (T, state_dim)
    actions_list,  # list of (T, action_dim)
    q1,
    q2,
    goal=np.array([0.0, 0.0]),
    num_bins=32,
    use_model_based=False,
    actor=None,
    dynamics=None,
    device="cpu",
    save_path=None,
    return_stats=False,
):
    """
    For all trajectories and all timesteps:
      - compute Fisher information (||∂Q/∂a||^2)
      - x-axis: distance to goal
      - y-axis: Fisher information

    Plot:
      - scatter of all samples
      - distance-binned mean + median
      - ±1 std band
    """

    all_distances = []
    all_fi = []

    # --------------------------------------------------
    # Collect FI samples
    # --------------------------------------------------
    for traj, actions in zip(trajectories, actions_list):
        T = min(len(traj), len(actions))

        s = torch.as_tensor(traj[:T], dtype=torch.float32, device=device)
        a = torch.as_tensor(actions[:T], dtype=torch.float32, device=device)

        distances = np.linalg.norm(traj[:T] - goal, axis=1)

        for t in range(T):
            st = s[t:t+1]
            at = a[t:t+1]

            if use_model_based:
                assert actor is not None and dynamics is not None
                fi = sac.model_based_q_fisher(
                    actor, q1, q2, dynamics, st, at
                )
            else:
                fi = sac.q_fisher(q1, q2, st, at)

            fi_val = fi.item()
            if np.isnan(fi_val) or np.isinf(fi_val):
                continue

            all_distances.append(distances[t])
            all_fi.append(fi_val)

    all_distances = np.asarray(all_distances)
    all_fi = np.asarray(all_fi)

    if len(all_fi) == 0:
        print("No valid Fisher information values.")
        return

    # --------------------------------------------------
    # Distance binning (based on valid FI samples)
    # --------------------------------------------------
    d_min, d_max = all_distances.min(), all_distances.max()
    bin_edges = np.linspace(d_min, d_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_means = []
    bin_medians = []
    bin_stds = []

    for i in range(num_bins):
        mask = (all_distances >= bin_edges[i]) & (all_distances < bin_edges[i + 1])
        vals = all_fi[mask]

        if len(vals) == 0:
            bin_means.append(np.nan)
            bin_medians.append(np.nan)
            bin_stds.append(np.nan)
        else:
            bin_means.append(np.mean(vals))
            bin_medians.append(np.median(vals))
            bin_stds.append(np.std(vals))

    bin_means = np.asarray(bin_means)
    bin_medians = np.asarray(bin_medians)
    bin_stds = np.asarray(bin_stds)

    if return_stats:
        return {
            "bin_centers": bin_centers,
            "means": bin_means,
            "medians": bin_medians,
            "stds": bin_stds,
        }

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    # Raw samples
    plt.scatter(
        all_distances,
        all_fi,
        s=10,
        alpha=0.35,
        edgecolors="none",
        label="FI samples",
    )

    # Mean / median trend
    plt.plot(
        bin_centers,
        bin_means,
        color="red",
        lw=2,
        label="Mean FI",
    )

    plt.plot(
        bin_centers,
        bin_medians,
        color="orange",
        lw=2,
        linestyle="--",
        label="Median FI",
    )

    # Std band
    plt.fill_between(
        bin_centers,
        bin_means - bin_stds,
        bin_means + bin_stds,
        color="red",
        alpha=0.25,
        label="±1 std",
    )

    plt.xlabel("Distance to goal")
    plt.ylabel("Fisher information (||∂Q/∂a||²)")
    title = "Model-based FI vs distance" if use_model_based else "Action FI vs distance"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        if use_model_based:
            fname = "fisher_vs_distance_model_based.png"
        else:
            fname = "fisher_vs_distance_action_based.png"

        plt.savefig(os.path.join(save_path, fname))
        plt.close()
    else:
        plt.show()


def plot_fisher_difference(
    mf_stats,  # model-free stats
    mb_stats,  # model-based stats
    title=None,
    save_path=None,
    fname="fisher_difference_mean_median.png",
):
    """
    Plot both mean and median FI differences:
        delta_mean(d)   = mean_MB(d)   - mean_MF(d)
        delta_median(d) = median_MB(d) - median_MF(d)
    """

    centers = mf_stats["bin_centers"]

    mf_mean, mb_mean = mf_stats["means"], mb_stats["means"]
    mf_med,  mb_med  = mf_stats["medians"], mb_stats["medians"]
    delta_mean   = mb_mean - mf_mean
    delta_median = mb_med  - mf_med

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.axhline(0.0, color="black", lw=1, alpha=0.8)

    # Median (main signal)
    plt.plot(
        centers,
        delta_median,
        lw=2.5,
        label="Median (MB - MF)",
    )

    # Mean (reference)
    plt.plot(
        centers,
        delta_mean,
        lw=2,
        linestyle="--",
        alpha=0.8,
        label="Mean (MB - MF)",
    )

    # Shading based on median dominance
    plt.fill_between(
        centers, delta_median, 0,
        where=(delta_median > 0),
        alpha=0.25,
        label="Model-based higher (median)",
    )
    plt.fill_between(
        centers, delta_median, 0,
        where=(delta_median < 0),
        alpha=0.25,
        label="Model-free higher (median)",
    )

    plt.xlabel("Distance to goal")
    plt.ylabel("FI difference (MB - MF)")
    if title is None:
        title = "Mean & median Fisher dominance vs distance"
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, fname))
        plt.close()
    else:
        plt.show()


def plot_fisher_ratio(
    mf_stats,  # model-free stats dict
    mb_stats,  # model-based stats dict
    eps=1e-8,  # numerical stability
    log_ratio=False,   # True: plot log(MB/MF)
    title=None,
    save_path=None,
    fname="fisher_ratio_mean_median.png",
):
    """
    Plot Fisher information ratio:
        ratio_mean(d)   = mean_MB(d)   / mean_MF(d)
        ratio_median(d) = median_MB(d) / median_MF(d)

    Interpretation:
        > 1  : model-based dominates
        < 1  : model-free dominates
    """

    centers = mf_stats["bin_centers"]

    mf_mean, mb_mean = mf_stats["means"],   mb_stats["means"]
    mf_med,  mb_med  = mf_stats["medians"], mb_stats["medians"]

    # --------------------------------------------------
    # Ratio
    # --------------------------------------------------
    ratio_mean   = mb_mean / (mf_mean + eps)
    ratio_median = mb_med  / (mf_med  + eps)

    if log_ratio:
        ratio_mean   = np.log(ratio_mean)
        ratio_median = np.log(ratio_median)
        ref_line = 0.0
        ylabel = "log FI ratio  log(MB / MF)"
    else:
        ref_line = 1.0
        ylabel = "FI ratio  (MB / MF)"

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.axhline(ref_line, color="black", lw=1, alpha=0.8)

    # Median (main)
    plt.plot(
        centers,
        ratio_median,
        lw=2.5,
        label="Median ratio",
    )

    # Mean (reference)
    plt.plot(
        centers,
        ratio_mean,
        lw=2,
        linestyle="--",
        alpha=0.85,
        label="Mean ratio",
    )

    # Shading based on median dominance
    if not log_ratio:
        plt.fill_between(
            centers, ratio_median, ref_line,
            where=(ratio_median > ref_line),
            alpha=0.25,
            label="Model-based dominant (median)",
        )
        plt.fill_between(
            centers, ratio_median, ref_line,
            where=(ratio_median < ref_line),
            alpha=0.25,
            label="Model-free dominant (median)",
        )
    else:
        plt.fill_between(
            centers, ratio_median, ref_line,
            where=(ratio_median > 0),
            alpha=0.25,
            label="Model-based dominant (median)",
        )
        plt.fill_between(
            centers, ratio_median, ref_line,
            where=(ratio_median < 0),
            alpha=0.25,
            label="Model-free dominant (median)",
        )

    plt.xlabel("Distance to goal")
    plt.ylabel(ylabel)

    if title is None:
        title = "Fisher information dominance vs distance (ratio)"
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, fname))
        plt.close()
    else:
        plt.show()


def get_spectral_norms(model):
    norms = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_orig'):
                norms[f"{name}_orig"] = torch.linalg.norm(module.weight_orig.detach(), ord=2).item()
            
            norms[f"{name}_eff"] = torch.linalg.norm(module.weight.detach(), ord=2).item()
    return norms


def plot_spectral_norms(norm_history, save_path):
    plt.figure(figsize=(12, 6))
    
    for name, values in norm_history.items():
        linestyle = ':' if 'orig' in name else '-'
        linewidth = 1.5 if 'eff' in name else 1.0
        alpha = 0.9 if 'eff' in name else 0.6
        
        plt.plot(values, label=name, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    
    plt.xlabel("Step (x100)")
    plt.ylabel("Spectral Norm (L2)")
    plt.title("Spectral Norm Evolution")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "spectral_norm_evolution.png"), dpi=300)
    plt.close()