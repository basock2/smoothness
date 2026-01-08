import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

seed_list = [0, 1, 2, 3, 4]
root = "./experiment_results"

for seed in seed_list:
    mb_path = os.path.join(root, f"seed_{seed}", "mb_smoothness_vs_distance_bins.npz")
    mf_path = os.path.join(root, f"seed_{seed}", "mf_smoothness_vs_distance_bins.npz")

    if not (os.path.exists(mb_path) and os.path.exists(mf_path)):
        print(f"[skip] seed {seed}: missing file")
        continue

    mb = np.load(mb_path)
    mf = np.load(mf_path)

    plt.figure(figsize=(7, 5))

    plt.plot(
        mb["bin_centers"],
        mb["bin_medians"],
        lw=2,
        label="Model-based",
    )

    plt.plot(
        mf["bin_centers"],
        mf["bin_medians"],
        lw=2,
        linestyle="--",
        label="Model-free",
    )

    plt.xlabel("Distance to goal")
    plt.ylabel("Action smoothness")
    plt.title(f"Smoothness vs distance (seed {seed})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_dir = os.path.join(root, f"seed_{seed}")
    save_path = os.path.join(save_dir, "smoothness_vs_distance_mb_vs_mf.png")

    plt.savefig(save_path)
    plt.close()

    print(f"[saved] {save_path}")
