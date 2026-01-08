import numpy as np
import torch
import sac_meta_soft as sac

def train_model(env, num_episodes, batch_size, start_steps, updates_per_step, seed, 
          actor, q1, q2, q1_t, q2_t, dynamics, opt_a, opt_q1, opt_q2, opt_dynamics, buffer, device, 
          alpha=0.2, eps=0.95, k_eps=1.0):
    total_steps = 0
    env.reset(seed=seed)
    prev_model_loss = None
    eps_min, eps_max = 0.05, 0.95

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env.action_space.seed(ep_seed)
        s, _ = env.reset(seed=ep_seed)
        s = torch.tensor(s, dtype=torch.float32).to(device)

        ep_reward = 0
        episode_model_losses = []

        while True:
            # -------- action selection --------
            if total_steps < start_steps:
                a = torch.tensor(env.action_space.sample(),
                                dtype=torch.float32).to(device)
            else:
                with torch.no_grad():
                    a, _ = actor.sample(s.unsqueeze(0))  # add batch dim
                    a = a.squeeze(0)  # remove batch dim

            ns, r, done, trunc, _ = env.step(a.cpu().numpy())
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
                    info = sac.sac_update(
                        actor, q1, q2, q1_t, q2_t, dynamics,
                        opt_a, opt_q1, opt_q2, opt_dynamics,
                        batch, alpha=alpha, eps=eps,
                    )
                    if info is not None and "model_loss" in info:
                        episode_model_losses.append(info["model_loss"])

            if done or trunc:
                break

        if len(episode_model_losses) > 0:
            curr_model_loss = np.mean(episode_model_losses)
            if prev_model_loss is not None:
                delta_L = (prev_model_loss - curr_model_loss) / (prev_model_loss + 1e-8)
                eps = eps * np.exp(-k_eps * delta_L)
                eps = float(np.clip(eps, eps_min, eps_max))
            prev_model_loss = curr_model_loss

        print(
            f"[Train] Ep {ep:03d} | Return {ep_reward:.2f} "
            f"| ModelLoss {curr_model_loss:.4f} | eps {eps:.3f}"
        )