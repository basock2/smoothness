import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import core as c
import sac
import numpy as np
import torch
import torch.optim as optim
import random
import gymnasium as gym

# --------initialization--------
seed = 28
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = sac.Actor(state_dim=state_dim, action_dim=action_dim).to(device)
q1 = sac.QNet(state_dim=state_dim, action_dim=action_dim).to(device)
q2 = sac.QNet(state_dim=state_dim, action_dim=action_dim).to(device)
q1_t = sac.QNet(state_dim=state_dim, action_dim=action_dim).to(device)
q2_t = sac.QNet(state_dim=state_dim, action_dim=action_dim).to(device)
dynamics = sac.DynamicsModel(state_dim=state_dim, action_dim=action_dim).to(device)

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
            actor, q1, q2, q1_t, q2_t, dynamics, opt_a, opt_q1, opt_q2, opt_dynamics, buffer, device)

# --------test--------
traj, action = sac.test_policy(env, actor, device, episodes=100)

# --------plot--------
c.smoothness_score(action)