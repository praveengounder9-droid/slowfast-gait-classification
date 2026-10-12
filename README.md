# Effect of Fine-Tuning Depth on Transfer Learning for Video-Based Gait Classification

This repository implements a comparative study investigating how progressive transfer learning unfreezing scopes affect classification boundaries and GPU computational metrics. Using **Meta's SlowFast-R50 architecture**, the framework evaluates three behavioral walking characteristics: **Regular**, **Limping**, and **Shuffling**.

## 📊 Dataset & Baseline Constraints
* **Source Dataset**: WeightGait Dataset (University of Edinburgh, School of Informatics)
* **Dataset Link**: https://doi.org
* **Input Specifications**: Uniform downsampled video tensor blocks `[Batch, Channels, Frames, Height, Width]` structured into a dual-pathway configuration:
  * **Slow Pathway**: Spatial posture stream tracking **8 sequential frames**
  * **Fast Pathway**: Temporal high-frequency motion tracking **32 sequential frames**
* **Environmental Control Factors**: Held constant across all experimental variations:
  * **Data Split**: 599 Training, 150 Validation, 149 Testing tensors
  * **Optimizer**: AdamW (`weight_decay=1e-4`)
  * **Objective Function**: Cross-Entropy Loss
  * **Execution Space**: Local execution on an NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
  * **Early Stopping**: Validation check tolerance set to **6 epochs** patience limit

## 🧬 Architectural Mapping
The pipeline utilizes a 7-block architecture sequencing model layer indexes from 0 to 6. The unfreezing configurations target the following structural layers:
* **Index 0**: Initial 3D Spatial-Temporal Stem
* **Indices 1-3**: ResNet Convolutional Stages 2, 3, and 4
* **Index 4**: ResNet Stage 5 (*The true last convolutional block*)
* **Index 5**: `PoolConcatPathway` (*Non-parametric structural connector - 0 weights*)
* **Index 6 / `[-1]`**: `ResNetBasicHead` (*Houses the modified 3-class linear classifier*)

---

## 📈 Experimental Metrics Matrix

| Metric Performance | Exp 1: Feature Extraction | Exp 2: Last Block Fine-Tuning | Exp 3: Partial Fine-Tuning | Exp 4: Full Fine-Tuning |
| :--- | :---: | :---: | :---: | :---: |
| **Trainable Layer Scope** | `blocks.6.proj` | `blocks.4`, `blocks.6.proj` | `blocks.3`, `blocks.4`, `blocks.6.proj` | `model.parameters()` |
| **Trainable Parameters** | **6,915** | **21,450,243** | **32,049,667** | **33,651,403** |
| **Learning Rate** | `3e-4` | `3e-4` | `3e-4` | **`1e-5`** *(Defensive Drop)* |
| **Best Target Epoch** | Epoch 8 | Epoch 12 | Epoch 5 | Epoch 9 |
| **Best Validation Loss** | 0.7316 | 0.0405 | 0.0674 | 0.0536 |
| **Final Test Accuracy** | **78.52%** | **95.30%** | 👑 **97.99%** | **95.30%** |
| **Macro F1-Score** | 0.7811 | 0.9523 | 👑 **0.9796** | **0.9537** |
| **GPU Compute Time** | ~1.5 Minutes | ~18.5 Minutes | **24.89 Minutes** | **220.28 Minutes** *(3.67 Hours)* |

---

## 🔍 Core Structural Insights

1. **The Performance Peak (Experiment 3 Winner)**: **Experiment 3 (Partial Fine-Tuning)** stands as the definitive project champion, securing a peak classification rate of **97.99%**. Unfreezing the 6 deep bottleneck blocks within `blocks.3` (Stage 4) expanded mid-level structural capabilities, successfully breaking the baseline model's *Limping* classification errors.
2. **Diminishing Returns (Experiment 4 Mismatch)**: Fully unfreezing the network in Experiment 4 caused training runtime to skyrocket to **220.28 minutes** (a **785% computational spike** over Experiment 3). However, accuracy declined by **-2.69%**. Unfreezing early stem layers introduced high-frequency background noise, which caused the model to over-flag healthy walks as subtle limps.
3. **The Parameter Verdict**: For custom tasks using compact video datasets, **Partial Fine-Tuning (Stages 4, 5, and Head)** provides the optimal balance. It updates the high-level convolutional kernels needed to track structural gait abnormalities while keeping early layers frozen to prevent overfitting.
