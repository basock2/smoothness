import numpy as np
import torch

def test_policy(env, actor, device, episodes=5):
    actor.eval()
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
                a, _ = actor.sample(s.unsqueeze(0))
                a = a.squeeze(0)

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

    actor.train()
    print(f"[Test] Mean Return {np.mean(total_r_list):.2f}")
    return traj_list, action_list
