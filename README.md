# bipedalwalker-v3-normal-and-hardcore-rl
# Research-Guided RL: Solving BipedalWalkerHardcore-v3

**Author:** Meicheng Wang  
**Status:** Solved (Peak Score: 281.8)  
**License:** MIT  

## 📌 Project Overview
This repository contains the implementation and experimental results for solving the `BipedalWalker-v3` and `BipedalWalkerHardcore-v3` environment using **Proximal Policy Optimization** and **Truncated Quantile Critics (TQC)**.

The project explores the transition from on-policy methods (PPO) to off-policy distributional reinforcement learning (TQC) to handle the stochastic obstacles of the Hardcore terrain. While tuned PPO agents achieved satisfying performance on normal-difficulty terrain, they failed to generalize to the rugged features of Hardcore mode. By leveraging TQC's distributional value estimation and truncation mechanism, I successfully mitigated overestimation bias and almost solved the environment by approaching 300 reward.

## 🚀 Ablation Studies & Results

We performed a systematic ablation study to isolate the impact of normalization, network architecture, and exploration strategies.

| Experiment | Environment | Key Modification | Timesteps | Best Mean Reward | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **basic_ppo_final** | Normal-v3 | Default | 2M | 296.9 | Almost Solved |
| **Improved_ppo_v2** | Normal-v3 | VecNorm, Lin.LR | 2M | **344.4** | **Solved** |
| **Improved ppo v2** | Hardcore-v3 | Baseline Transfer | 3M | 21.6 | Failed |
| **sota_ppo_v3** | Hardcore-v3 | gSDE, Decoupled Net, Large Net | 60k* | -122.9 | Failed |
| **sota_ppo_v3_1** | Hardcore-v3 | -gSDE (Standard Noise) | 3M | 112.9 | Converging (Stuck) |
| **Tqc_v3_hardcore** | Hardcore-v3 | TQC (Off-Policy) | 3M | 274.2 | Almost Solved |
| **Tqc_v3_hardcore_finetune** | Hardcore-v3 | Continued Training | 4.8M (Total) | **281.8** | **Almost Solved** |
| **Tqc_v3_hardcore_5M** | Hardcore-v3 | Long Training | 5M | 249.2 | Failed (Regression) |

*\*Note: sota_ppo_v3 was interrupted early due to evident instability/failure.*

## 📂 Repository Structure

* **`finalone.ipynb`**: The main Jupyter notebook containing all code for:
    * **PPO Experiments:** Baseline, Tuned (v2), and Hybrid (v3.1).
    * **TQC Experiments:** The final solution using SB3 Contrib.
    * **Visualization:** Code to reproduce the learning curves and tables.
* **`models/`**: Pre-trained agent checkpoints.
    * `improved_ppo_v2`: Solved Normal mode agent.
    * `tqc_v3_hardcore_finetune`: Best performing Hardcore agent.
    * *(Note: Folders ending in `_ultimate`, `_zoo_tuned` are interrupted/failed runs due to internet and unimportant issues and please ignore them).*
* **`logs/`**: TensorBoard/CSV logs used to generate the ablation graphs.
* **`videos/`**: MP4 recordings of the agents' performance.

## 🛠️ Installation & Reproduction

To reproduce these results, you can run the `finalone.ipynb` notebook in Google Colab (recommended for GPU access).

```bash
# Install dependencies
pip install -r requirements.txt
