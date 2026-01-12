import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, s):
        mu, std = self(s)
        eps = torch.randn_like(std)
        z = mu + eps * std
        a = torch.tanh(z)

        logp = (
            -0.5 * ((z - mu) / std).pow(2)
            - torch.log(std)
            - 0.5 * np.log(2 * np.pi)
        ).sum(-1)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)

        return a, logp


class QNet(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x).squeeze(-1)


def sac_update(
    actor, q1, q2, q1_t, q2_t, dynamics,
    opt_a, opt_q1, opt_q2, opt_dynamics,
    batch, alpha=0.2, gamma=0.99, tau=0.005,
):
    s, a, r, ns, d = batch

    # ----- Dynamics -----
    pred_ns = dynamics(s, a)
    dynamics_loss = F.mse_loss(pred_ns, ns)
    opt_dynamics.zero_grad(); dynamics_loss.backward(); opt_dynamics.step()

    # ----- Critic -----
    with torch.no_grad():
        r = r.squeeze(-1); d = d.squeeze(-1)
        na, nlogp = actor.sample(ns)
        tq = torch.min(q1_t(ns, na), q2_t(ns, na))
        target = r + gamma * (1 - d) * (tq - alpha * nlogp)

    q1_loss = F.mse_loss(q1(s, a), target)
    q2_loss = F.mse_loss(q2(s, a), target)

    opt_q1.zero_grad(); q1_loss.backward(); opt_q1.step()
    opt_q2.zero_grad(); q2_loss.backward(); opt_q2.step()

    # ----- Actor -----
    a, logp = actor.sample(s)
    pred_ns = dynamics(s, a)
    na, nlogp = actor.sample(pred_ns)
    q_val = torch.min(q1(pred_ns, na), q2(pred_ns, na))
    target = r + gamma * q_val

    actor_loss = (alpha * logp - target).mean()

    opt_a.zero_grad()
    actor_loss.backward()
    opt_a.step()

    # ----- Target update -----
    for p, pt in zip(q1.parameters(), q1_t.parameters()):
        pt.data.mul_(1 - tau).add_(tau * p.data)
    for p, pt in zip(q2.parameters(), q2_t.parameters()):
        pt.data.mul_(1 - tau).add_(tau * p.data)

    return actor_loss.item()

def q_state_gradient_field(actor, q, xs, ys):
    grid = torch.stack(
        torch.meshgrid(xs, ys, indexing="ij"),
        dim=-1
    ).reshape(-1, 2).requires_grad_(True)

    a, _ = actor.sample(grid)
    q_val = q(grid, a)

    grad_s = torch.autograd.grad(q_val.sum(), grid)[0]
    return grid.detach(), grad_s.detach()

def q_action_gradient_field(actor, q, trajs, device):
    grad_a = []
    if isinstance(trajs, np.ndarray):
        trajs = [trajs]
    for i, traj in enumerate(trajs):
        traj = torch.FloatTensor(traj).to(device)
        a, _ = actor.sample(traj)
        a = a.detach().requires_grad_(True)
        q_val = q(traj, a)
        grad_a.append(torch.autograd.grad(q_val.sum(), a)[0].detach())
    return trajs, grad_a

def q_model_based_action_gradient_field(actor, q1, q2, dynamics, trajs, device):
    grad_a = []
    if isinstance(trajs, np.ndarray):
        trajs = [trajs]
    for i, traj in enumerate(trajs):
        traj = torch.FloatTensor(traj).to(device)
        a, _ = actor.sample(traj)
        a = a.detach().requires_grad_(True)
        ns = dynamics.forward(traj, a)
        mu, _ = actor.forward(ns)
        na = torch.tanh(mu)
        qval = torch.min(q1(ns, na), q2(ns, na))
        model_action_grad = torch.autograd.grad(
            outputs=qval.sum(),
            inputs=a
        )[0]
        grad_a.append(model_action_grad.detach())
    return trajs, grad_a


# buffer implementation
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, s, a, r, ns, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, ns, d)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(torch.stack, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)


class DynamicsModel(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ds = nn.Linear(hidden, state_dim)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta_s = self.ds(x)
        next_state_pred = s + delta_s
        return next_state_pred
    
    def get_model_based_action_grad(self, s, a, actor, q1, q2):
        a = a.detach().clone()
        a.requires_grad = True
        ns = self.forward(s, a)
        mu, _ = actor.forward(ns)
        na = torch.tanh(mu)
        qval = torch.min(q1(ns, na), q2(ns, na))
        model_action_grad = torch.autograd.grad(
            outputs=qval.sum(),
            inputs=a
        )[0]
        return model_action_grad


def q_fisher(q1, q2, s, a):
    a = a.detach().clone()
    a.requires_grad = True
    q_val = torch.min(q1(s, a), q2(s, a))
    q_grad = torch.autograd.grad(
        outputs=q_val.sum(), 
        inputs=a, 
        create_graph=False, 
        retain_graph=False
    )[0]
    FI = torch.sum(q_grad ** 2, dim=-1)

    return FI


def model_based_q_fisher(actor, q1, q2, dynamics, s, a):
    a = a.detach().clone()
    a.requires_grad = True
    ns = dynamics.forward(s, a)
    mu, _ = actor.forward(ns)
    na = torch.tanh(mu)
    qval = torch.min(q1(ns, na), q2(ns, na))
    model_q_grad = torch.autograd.grad(
        outputs=qval.sum(),
        inputs=a,
        create_graph=False, 
        retain_graph=False
    )[0]
    FI = torch.sum(model_q_grad ** 2, dim=-1)

    return FI


def actor_mu(actor, s):
    mu, _ = actor(s)
    return torch.tanh(mu)