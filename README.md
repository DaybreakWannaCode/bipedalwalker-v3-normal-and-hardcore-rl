# Research-Guided RL: Solving BipedalWalker-v3

**Author:** Meicheng Wang

**Status:** Normal (Peak Mean Reward: 344.4) Hardcore (Peak Mean Reward: 281.3)

## 📌 Project Overview

This repository contains the implementation and experimental results for solving the `BipedalWalker-v3` and `BipedalWalkerHardcore-v3` environments using **Proximal Policy Optimization (PPO)** and **Truncated Quantile Critics (TQC)**.

The project explores the transition from on-policy methods (PPO) to off-policy distributional reinforcement learning (TQC) to handle the stochastic obstacles of the Hardcore terrain. While tuned PPO agents achieved satisfying performance on normal-difficulty terrain, they failed to generalize to the rugged features of Hardcore mode. By leveraging TQC's distributional value estimation and truncation mechanism, I successfully mitigated overestimation bias and achieved a score approaching 300.


| Agent Name | GitHub Name (Raw Model) | Environment | Key Modification | Timesteps | Best Mean Reward | Result |
| --- | --- | --- | --- | --- | --- | --- |
| **PPO-Baseline** | `basic_ppo_final` | Normal-v3 | Default | 2M | 296.9 | Almost Solved |
| **PPO-Tuned** | `improved_ppo_v2` | Normal-v3 | VecNorm, Lin.LR | 2M | **344.4** | **Solved** |
| **PPO-Tuned** | `improved_ppo_v2` | Hardcore-v3 | Baseline Transfer | 3M | 21.6 | Failed |
| **PPO-Adv-gSDE** | `sota_ppo_v3_hardcore` | Hardcore-v3 | gSDE, Large Net | 60k* | -122.9 | Failed |
| **PPO-Advanced** | `sota_ppo_v3_1_hardcore` | Hardcore-v3 | -gSDE (Standard Noise) | 3M | 112.9 | Converging (Stuck) |
| **TQC-Best** | `tqc_v3_hardcore_rerun...1928` | Hardcore-v3 | TQC (Off-Policy) | 3M | 274.2 | Almost Solved |
| **TQC-Resume** | `...1928_resume_3M` | Hardcore-v3 | Continued Training | 4.8M | **281.3** | **Solved** |
| **TQC-5M** | `tqc_v3_hardcore_5M` | Hardcore-v3 | Long Training (1 Run) | 5M | 249.2 | Regression |

**Note: PPO-Adv-gSDE was interrupted early due to evident instability/failure.*

## 📂 Repository Structure

### **Core Notebooks**

* **`finalone.ipynb`**: The **Master Notebook**. Contains the code for:
* Training the Baseline and Best agents (PPO-Tuned, TQC-Best).

* **`ablationStudy1.ipynb`**: **Secondary Session**. Used to run parallel ablation experiments without blocking the main training loop.
* **`ablationStudy2.ipynb`**: **Tertiary Session**.  Used to run parallel ablation experiments without blocking the main training loop.


### **Model Directory Mapping (`models/`)**

*To ensure reproducibility, use this table to map the "Paper Designation" to the raw filenames in this repo.*

| Paper Designation | Raw Directory Name | Description |
| --- | --- | --- |
| **PPO-Baseline** | `basic_ppo_final` | Initial PPO attempt (Normal mode). |
| **PPO-Tuned** | `improved_ppo_v2` | **Best Normal Agent**. Added `VecNormalize`. |
| **PPO-Adv-gSDE** | `sota_ppo_v3_hardcore` | Failed experiment using gSDE. |
| **PPO-Advanced** | `sota_ppo_v3_1_hardcore` | **Best PPO Agent** (Hardcore). Decoupled, no gSDE. |
| **TQC-Best** | `tqc_v3_hardcore_rerun...1928` | **Primary Solved Agent** (0-3M steps). |
| **TQC-Resume** | `...1928_resume_3M` | **Peak Score Agent** (Resumed training to 281.3). |

*Note: Some folders stored in this repository are deprecated legacy runs or failed experiments caused by connection interruptions and other errors. They contain no useful information and the best way to navigate this repository is to find the successful agents listed above with their raw directory name.*

### **Other Directories**

* **`logs/`**: Raw TensorBoard/Monitor CSV logs used to generate the learning curves.
* **`videos/`**: MP4 recordings of the agents' performance (Best Normal & Best Hardcore runs).

## 🛠️ Installation & Reproduction

To reproduce these results, you can run the `finalone.ipynb` notebook in Google Colab (recommended for GPU access).

```bash
# Install dependencies (Gymnasium, Stable-Baselines3, Box2D)
pip install gymnasium[box2d] stable-baselines3 sb3-contrib shimmy pyvirtualdisplay

```
