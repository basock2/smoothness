import numpy as np
import torch
import sac

def train_model(env, num_episodes, batch_size, start_steps, updates_per_step, seed, 
          actor, q1, q2, q1_t, q2_t, dynamics, opt_a, opt_q1, opt_q2, opt_dynamics, buffer, device):
    total_steps = 0
    env.reset(seed=seed)
    for ep in range(num_episodes):
        ep_seed = seed + ep
        env.action_space.seed(ep_seed)
        s, _ = env.reset(seed=ep_seed)
        s = torch.tensor(s, dtype=torch.float32).to(device)

        ep_reward = 0

        while True:
            # -------- action selection --------
            if total_steps < start_steps:
                a = torch.tensor(env.action_space.sample(),
                                dtype=torch.float32).to(device)
            else:
                with torch.no_grad():
                    a, _ = actor.sample(s.unsqueeze(0))  # add batch dim
                    a = a.squeeze(0)  # remove batch dim

            ns, r, done, trunc, _ = env.step(scale_action(a, env).cpu().numpy())
            ns = torch.tensor(ns, dtype=torch.float32).to(device)
            r = torch.tensor([r], dtype=torch.float32).to(device)
            d = torch.tensor([done], dtype=torch.float32).to(device)

            buffer.push(
                s.detach().cpu(), 
                a.detach().cpu(), 
                r.detach().cpu(), 
                ns.detach().cpu(), 
                d.detach().cpu()
            )

            s = ns
            ep_reward += r.item()
            total_steps += 1

            # -------- update --------
            if len(buffer) > batch_size:
                for _ in range(updates_per_step):
                    batch = buffer.sample(batch_size)
                    batch = [x.to(device) for x in batch]
                    sac.sac_update(
                        actor, q1, q2, q1_t, q2_t, dynamics,
                        opt_a, opt_q1, opt_q2, opt_dynamics,
                        batch,
                    )

            if done or trunc:
                break

        print(f"[Train] Episode {ep:03d} | Return {ep_reward:.2f}")


def scale_action(a, env):
    low = torch.as_tensor(env.action_space.low, device=a.device)
    high = torch.as_tensor(env.action_space.high, device=a.device)
    return low + (a + 1.0) * 0.5 * (high - low)