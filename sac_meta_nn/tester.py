import numpy as np
import torch
import sac_meta_nn as sac

def test_policy(env, actor_mf, actor_mb, lambda_net, device, episodes=5):
    actor_mf.eval()
    actor_mb.eval()
    lambda_net.eval()
    traj_list = []
    action_list = []
    total_r_list = []
    for ep in range(episodes):
        s, _ = env.reset()
        s = torch.tensor(s, dtype=torch.float32).to(device)

        traj = [s.cpu().numpy()]
        action = []
        total_r = 0

        while True:
            with torch.no_grad():
                lamb = lambda_net(s.unsqueeze(0))  # add batch dim
                a_mf = sac.actor_mu(actor_mf, s.unsqueeze(0))
                a_mb = sac.actor_mu(actor_mb, s.unsqueeze(0))
                a = torch.where(lamb < 0.5, a_mb, a_mf)
                a = a.squeeze(0)  # remove batch dim

            ns, r, done, trunc, _ = env.step(a.cpu().numpy())
            ns = torch.tensor(ns, dtype=torch.float32).to(device)

            traj.append(ns.cpu().numpy())
            action.append(a.cpu().numpy())
            total_r += r

            s = ns
            if done or trunc:
                break

        traj = np.array(traj)
        traj_list.append(traj)
        action = np.array(action)
        action_list.append(action)
        print(f"[Test] Episode {ep} | Return {total_r:.2f}")
        total_r_list.append(total_r)

    actor_mf.train()
    actor_mb.train()
    lambda_net.train()
    print(f"[Test] Mean Return {np.mean(total_r_list):.2f}")
    return traj_list, action_list
