import re
import matplotlib.pyplot as plt


def parse_log(filepath):
    """Parse step and critic/rewards/mean from training log."""
    steps = []
    rewards = []
    with open(filepath, "r") as f:
        for line in f:
            m = re.search(r"step:(\d+)", line)
            if not m:
                continue
            step = int(m.group(1))
            m2 = re.search(r"critic/score/mean:([\d.]+)", line)
            if not m2:
                continue
            reward = float(m2.group(1))
            steps.append(step)
            rewards.append(reward)
    return steps, rewards


random_steps, random_rewards = parse_log("countdown_random_baseline.log")
baseline_steps, baseline_rewards = parse_log("countdown_baseline.log")
ppl_steps, ppl_rewards = parse_log("countdown_edco.log")

plt.figure(figsize=(8, 5))
plt.plot(random_steps, random_rewards, label="Random", color="#7f7f7f", linewidth=2,)
plt.plot(baseline_steps, baseline_rewards, label="SEC Bandit", color="#1f77b4", linewidth=2)
plt.plot(ppl_steps, ppl_rewards, label="PGS", color="#d62728", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Reward (mean)")
plt.title("Step vs Reward: Random vs SEC Bandit vs PGS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scores_comparison_countdown.png", dpi=150)
plt.show()
