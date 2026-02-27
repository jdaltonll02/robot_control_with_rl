# Submission — RL for Robot Control Technical Assessment

## Candidate Information

- **Name:**
- **Email:**
- **GitHub Username:**
- **Date Submitted:**

---

## Summary

_In 3-5 sentences, summarize what you accomplished and your key findings. Highlight your HER implementation and the impact it had on performance._



---

## Links

- **Wandb Project:** `https://wandb.ai/<your-username>/rl-fetch-push`
- **Checkpoint (if hosted):** _(link or "local in repo")_

---

## Environment

- **GPU(s) used:** _(or CPU-only)_
- **Python version:**
- **PyTorch version:**
- **Gymnasium-Robotics version:**
- **OS / Platform:** (e.g., Google Colab, Kaggle, local)
- **Total compute time (approx):**

---

## Checklist

_Mark each item with [x] when complete._

### Part A: Setup, Baselines & HER
- [ ] Environment set up and verified (Gymnasium-Robotics + MuJoCo)
- [ ] Wandb logging enabled — training curves visible
- [ ] SAC baseline trained without HER — results saved
- [ ] HER implemented from scratch in `scripts/her_replay_buffer.py`
- [ ] SAC+HER trained — success rate significantly above baseline
- [ ] All runs logged to Wandb

### Part B: Reward Engineering & Algorithm Comparison
- [ ] Designed and implemented >= 2 custom reward functions in `fetch_push_env.py`
- [ ] Custom rewards work with HER (`compute_reward_static` implemented)
- [ ] Trained SAC+HER with each custom reward, results saved
- [ ] Compared SAC+HER vs DDPG+HER with best reward, results saved
- [ ] Videos recorded showing trained agent behavior

### Part C: Robustness & Report
- [ ] Best policy evaluated under domain shifts (mass, friction, size variations)
- [ ] Trained a domain-randomized policy with HER, compared vs nominal
- [ ] `REPORT.pdf` — paper-format technical report (4-8 pages)
- [ ] Report includes: HER analysis, reward design rationale, SAC vs DDPG analysis
- [ ] Report includes Wandb figures and video analysis
- [ ] Wandb project set to public (or shared with reviewer)

### Final
- [ ] Git history shows iterative progress
- [ ] Repository follows the required structure
- [ ] All scripts are reproducible (seeds, exact commands)

---

## Results Summary

### HER Impact

| Experiment | Success Rate | Mean Return |
|-----------|:------------:|:-----------:|
| SAC (no HER) | | |
| SAC + HER (sparse) | | |

### Reward Comparison (all with SAC + HER)

| Reward Type | Success Rate | Mean Return | Mean Energy |
|-------------|:------------:|:-----------:|:-----------:|
| sparse | | | |
| dense_basic | | | |
| _your_reward_1_ | | | |
| _your_reward_2_ | | | |

### Algorithm Comparison (best reward + HER)

| Metric | SAC + HER | DDPG + HER |
|--------|:---------:|:----------:|
| Success Rate | | |
| Mean Return | | |
| Mean Episode Length | | |
| Training Timesteps | | |
| Training Time (min) | | |

### Robustness (best policy under domain shifts)

| Mass | Friction | Nominal-trained | DR-trained |
|:----:|:--------:|:---------------:|:----------:|
| 0.5x | 1.0x | | |
| 1.0x | 1.0x | | |
| 1.5x | 1.0x | | |
| 2.0x | 1.0x | | |
| 1.0x | 0.5x | | |
| 1.0x | 1.5x | | |
| 1.0x | 2.0x | | |

---

## HER Implementation Details

- **Strategy:** _(e.g., "future" with k=4)_
- **Key design decisions:**
- **Challenges encountered:**

---

## Reward Design Details

- **Custom reward 1 name:** _e.g., dense_approach_
- **Formulation:** _Mathematical formula or pseudocode_
- **Motivation:** _Why did you design it this way?_
- **HER compatibility:** _How does this reward work with goal relabeling?_

- **Custom reward 2 name:** _e.g., progress_bonus_
- **Formulation:**
- **Motivation:**
- **HER compatibility:**

---

## Issues Encountered

1.
2.
3.

---

## Notes

_Additional context or observations for the reviewer._
