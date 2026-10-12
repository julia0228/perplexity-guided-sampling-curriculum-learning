import re
import matplotlib.pyplot as plt


def parse_val_scores(filepath):
    """Parse val/test_score from training log (recorded every 10 steps)."""
    steps = []
    easy = []
    medium = []
    hard = []
    very_hard = []

    with open(filepath, "r") as f:
        for line in f:
            m = re.search(r"step:(\d+)", line)
            if not m:
                continue
            step = int(m.group(1))
            m_easy = re.search(r"val/test_score/countdown-easy-test:([\d.]+)", line)
            if not m_easy:
                continue
            steps.append(step)
            easy.append(float(m_easy.group(1)))
            medium.append(float(re.search(r"val/test_score/countdown-medium-test:([\d.]+)", line).group(1)))
            hard.append(float(re.search(r"val/test_score/countdown-hard-test:([\d.]+)", line).group(1)))
            very_hard.append(float(re.search(r"val/test_score/countdown-very_hard-test:([\d.]+)", line).group(1)))

    return steps, {"easy": easy, "medium": medium, "hard": hard, "very_hard": very_hard}


random_steps, random_scores = parse_val_scores("countdown_random_baseline.log")
baseline_steps, baseline_scores = parse_val_scores("countdown_baseline.log")
ppl_steps, ppl_scores = parse_val_scores("countdown_edco.log")

difficulties = ["easy", "medium", "hard", "very_hard"]
colors = {"Random": "#7f7f7f", "SEC Bandit": "#1f77b4", "PGS": "#d62728"}
markers = {"Random": "D", "SEC Bandit": "o", "PGS": "s"}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, diff in zip(axes, difficulties):
    ax.plot(random_steps, random_scores[diff], label="Random",
            color=colors["Random"], marker="D", markersize=4, linewidth=1.5)
    ax.plot(baseline_steps, baseline_scores[diff], label="SEC Bandit",
            color=colors["SEC Bandit"], marker="o", markersize=4, linewidth=1.5)
    ax.plot(ppl_steps, ppl_scores[diff], label="PGS",
            color=colors["PGS"], marker="s", markersize=4, linewidth=1.5)
    ax.set_title(diff.replace("_", " ").title())
    ax.set_xlabel("Step")
    ax.set_ylabel("Test Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Val Test Score: Random vs SEC Bandit vs PGS", fontsize=14)
plt.tight_layout()
plt.savefig("val_score_comparison.png", dpi=150)
plt.show()
