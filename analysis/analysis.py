import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_rewards_from_file(filename):
    mean_rewards = []
    std_rewards = []
    with open(filename, "r") as f:
        text = f.read()
    mean_matches = re.findall(r"Mean Test Reward:\s*([-+]?\d*\.?\d+)", text)
    std_matches = re.findall(r"Std Reward:\s*([-+]?\d*\.?\d+)", text)
    mean_rewards = np.array(mean_matches, dtype=float)
    std_rewards = np.array(std_matches, dtype=float)
    mu, sigma = norm.fit(mean_rewards)
    x = np.linspace(mean_rewards.min() - 3 * sigma, mean_rewards.max() + 3 * sigma, 500)
    pdf = norm.pdf(x, mu, sigma)
    base_filename = os.path.splitext(filename)[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(np.arange(len(mean_rewards)), mean_rewards, yerr=std_rewards, fmt='o', color = "blue", markersize=6, capsize=2, alpha=0.6)
    ax.set_xlabel("Simulation")
    ax.set_ylabel("Mean Test Reward")
    ax.set_title("Mean Test Reward across simulations")
    plt.tight_layout()
    rewards_filename = f"../analysis/plots/{base_filename}_rewards.png"
    fig.savefig(rewards_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_rewards, bins=15, density=True, color = "blue", alpha=0.4, edgecolor="blue", label="Simulation data")
    ax.plot(x, pdf, linewidth=2, label=fr"Gaussian fit ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$)", color = "red")
    ax.axvline(mu, linestyle="--", linewidth=1.5, label=fr"$\mu={mu:.2f}$", color = "black")
    ax.set_xlabel("Mean Test Reward")
    ax.set_ylabel("Probability density")
    ax.set_title("Distribution of Mean Test Reward")
    ax.legend()
    plt.tight_layout()
    distribution_filename = f"../analysis/plots/{base_filename}_distribution.png"
    fig.savefig(distribution_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nGaussian fit:")
    print(f"Mean (μ) = {mu:.4f}")
    print(f"Std  (σ) = {sigma:.4f}")
    return mu, sigma
