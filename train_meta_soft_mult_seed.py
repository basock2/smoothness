import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import core as c
import sac_meta_soft as sac
import numpy as np
import torch
import torch.optim as optim
import random

# --------initialization--------
seed_list = [0, 1, 2, 3, 4]
base_result_dir = "./experiment_results"

for seed in seed_list:
    seed_dir = os.path.join(base_result_dir, f"seed_{seed}")
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = c.PointMass2D(goal=(0.0, 0.0))

    actor = sac.Actor().to(device)
    q1 = sac.QNet().to(device)
    q2 = sac.QNet().to(device)
    q1_t = sac.QNet().to(device)
    q2_t = sac.QNet().to(device)
    dynamics = sac.DynamicsModel().to(device)

    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    opt_a = optim.Adam(actor.parameters(), lr=3e-4)
    opt_q1 = optim.Adam(q1.parameters(), lr=3e-4)
    opt_q2 = optim.Adam(q2.parameters(), lr=3e-4)
    opt_dynamics = optim.Adam(dynamics.parameters(), lr=3e-4)

    buffer = sac.ReplayBuffer()

    # --------train--------
    num_episodes = 30
    batch_size = 128
    start_steps = 1000
    updates_per_step = 1

    sac.train_model(env, num_episodes, batch_size, start_steps, updates_per_step, seed, 
                actor, q1, q2, q1_t, q2_t, dynamics, opt_a, opt_q1, opt_q2, opt_dynamics, buffer, device, eps=0.5)

    # --------test--------
    traj, action = sac.test_policy(env, actor, device, episodes=100)

    # --------plot--------
    c.plot_qf(actor, q1, device, traj, save_path=seed_dir)
    c.plot_grad_a(actor, q1, device, traj, save_path=seed_dir)
    c.plot_grad_a_target(actor, q1, q2, dynamics, device, traj, save_path=seed_dir)
    c.smoothness_score(action)
    c.plot_smoothness_vs_distance(traj, action, save_path=seed_dir, save_data=False)
    mf_stats = c.plot_fisher_vs_distance(traj, action, q1, q2, return_stats=True)
    mb_stats = c.plot_fisher_vs_distance(traj, action, q1, q2, use_model_based=True, actor=actor, dynamics=dynamics, return_stats=True)
    c.plot_fisher_ratio(mf_stats, mb_stats, log_ratio=True, save_path=seed_dir)
    c.plot_fisher_vs_distance(traj, action, q1, q2, save_path=seed_dir)
    c.plot_fisher_vs_distance(traj, action, q1, q2, use_model_based=True, actor=actor, dynamics=dynamics, save_path=seed_dir)