# LTE Interference Classification via Semi-Supervised Teacher–Student Approach with CNN and ResNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch%20%7C%20TensorFlow-orange)](https://pytorch.org/)

This repository contains the official implementation, experimental notebooks, and results for the paper: **"LTE Interference Classification via Semi-Supervised Teacher–Student Approach with CNN and ResNet"**.

**Authors:** Jorge Luis Zegarra Guardamino, Percy Maldonado-Quispe.  
*School of Computer Science, National University of San Agustín de Arequipa (UNSA).*

---

## 📝 Abstract

This study proposes a pragmatic semi-supervised pipeline for classifying interference in the LTE uplink, designed to optimize expert time through a **Human-in-the-Loop (HITL)** scheme.

Unlike fully automated methods that risk error propagation in critical domains, our approach utilizes *Teacher* models to generate pseudo-labels, where only those with high confidence ($\ge 80\%$) are submitted for rapid expert validation.

**Key Findings:**
* There is a direct relationship between data density and the optimal architecture.
* **ResNet50** maximizes precision in carriers with high spectral complexity.
* **ResNet18** offers superior generalization in scenarios with sample scarcity.

## 📊 Key Results

The proposed approach successfully balances computational cost with operational safety:

| Scenario | Metric | Result |
| :--- | :--- | :--- |
| **Medium Density** | Macro F1-Score | **0.946** |
| **Medium Density** | Accuracy | **94.74%** |
| **High Volume** | F1-Score | 0.928 |

---

## 📂 Repository Structure

Based on the official implementation layout:

```text
├── MUESTRAS_ini/           # Original Raw Data (Source)
│   ├── Carrier_C1_675/     # Data per carrier
│   │   ├── ARM_ANCHO/      # Class folders
│   │   └── ...
│   └── ...
├── DATA_FOR_TRAINING/      # Processed Data (Split Train/Test)
│   └── Carrier_X/
│       ├── train_val/      # Training + Validation set (80%)
│       └── test/           # Held-out Test set (20%)
├── RESULTADOS/             # Experiment Outputs
│   ├── BENCHMARK/          # Resource usage stats
│   └── Carrier_X/          # Model checkpoints, Confusion Matrices, Logs
├── 00_data_prep.py         # Data splitting script
├── 01_train_teachers.ipynb # Teacher model training
├── 02_pseudo_labeling.ipynb# Pseudo-label generation (HITL)
├── 03_train_student.ipynb  # Student model training (Augmented)
├── 04_final_eval.py        # Official evaluation on Test Set
├── 05_benchmarking.py      # Resource benchmarking (FPS/Params)
├── 06_baseline_prep.ipynb  # Data prep for SVM/RF
├── 07_baselines.ipynb      # Classical ML Baselines (SVM/RF)
├── 08_run_ablation.ipynb   # Ablation studies execution
├── 09_gen_ablation_data.ipynb # Data generation for ablations
├── 10_data_stats.ipynb     # Data distribution analysis
├── config.py               # Global configuration and paths
├── models.py               # CNN & ResNet architectures
├── utils.py                # Helper functions
└── requirements.txt        # Dependencies
