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

- There is a direct relationship between data density and the optimal architecture.
- **ResNet50** maximizes precision in carriers with high spectral complexity.
- **ResNet18** offers superior generalization in scenarios with sample scarcity.

## 📊 Key Results

The proposed approach successfully balances computational cost with operational safety:

| Scenario         | Metric          | Result    |
| :--------------- | :-------------- | :-------- |
| **Medium Density** | Macro F1-Score | **0.946** |
| **Medium Density** | Accuracy        | **94.74%** |
| **High Volume**    | F1-Score        | 0.928     |

---

## 📂 Repository Structure

```text
├── MUESTRAS_ini/           # Original Raw Data (Source)
│   ├── Carrier_C1_675/     # Data per carrier
│   │   ├── ARM_ANCHO/      # Class folders
│   │   └── ...
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
├── 08_gen_ablation_data.ipynb # Data generation for ablations
├── 09_run_ablation.ipynb   # Ablation studies execution
├── 10_data_stats.ipynb     # Data distribution analysis
├── config.py               # Global configuration and paths
├── models.py               # CNN & ResNet architectures
├── utils.py                # Helper functions
└── requirements.txt        # Dependencies
````

---

## ⚙️ Execution Pipeline (Step-by-Step)

Follow these steps sequentially to replicate the Semi-Supervised Learning workflow.

### Prerequisites

Activate your Anaconda environment with GPU support before running scripts:

```bash
conda activate tf-gpu
```

*Ensure `config.py` is updated with the correct `CURRENT_CARRIER` and dataset paths.*

---

### Step 1: Data Preparation 🗂️

Splits the raw dataset (`MUESTRAS_ini`) into stratified training and validation sets (`DATA_FOR_TRAINING`).

* **Script:** `00_data_prep.py`

* **Command:**

  ```bash
  python 00_data_prep.py
  ```

* **Output:** Creates `DATA_FOR_TRAINING/` with `train_val` and `test` subdirectories.

---

### Step 2: Teacher Models Training 👨‍🏫

Trains the initial supervised baselines (Teachers) on labeled data.

* **Notebook:** `01_train_teachers.ipynb`
* **Action:** Run all cells.
* **Output:** Saves model checkpoints (`.pth`) in the `RESULTADOS/` directory and logs metrics to Excel.

---

### Step 3: Pseudo-Label Generation 🏷️

The best Teacher model infers labels for the unlabeled dataset based on the confidence threshold.

* **Notebook:** `02_pseudo_labeling.ipynb`
* **Action:** Run all cells.
* **Output:** Generates the `PSEUDO/` directory containing class-sorted images.

---

### Step 4: Human-in-the-Loop Review (HITL) 🧑‍🔬

**Manual Step:** An expert must verify the pseudo-labels to ensure quality.

* **Action:** Navigate to the `PSEUDO/` directory and verify images. Move or delete incorrectly classified samples.

---

### Step 5: Student Model Training 🎓

Trains the Student model using the Augmented Dataset (Original Labeled + Verified Pseudo-labels).

* **Notebook:** `03_train_student.ipynb`
* **Action:** Update the `student_experiments` config within the notebook and run all cells.
* **Output:** Validates that the Student model outperforms the Teacher (higher F1-Score).

---

### Step 6: Final Evaluation 📊

Definitive performance assessment on the isolated Test Set.

* **Script:** `04_final_eval.py`

* **Command:**

  ```bash
  python 04_final_eval.py
  ```

* **Output:** Generates the final Confusion Matrix (`.png`) and Classification Report (`.txt`) in the `RESULTADOS/` folder.

---

## 🔬 Reproducibility Summary Table

| Order   | Script Name                  | Description                                                          | Key Output                   |
| :------ | :--------------------------- | :------------------------------------------------------------------- | :--------------------------- |
| **00**  | `00_data_prep.py`            | Splits raw data into stratified Training and Test sets.              | `DATA_FOR_TRAINING/`         |
| **01**  | `01_train_teachers.ipynb`    | Trains initial Teacher models on labeled data.                       | Teacher Checkpoints          |
| **02**  | `02_pseudo_labeling.ipynb`   | Generates pseudo-labels for unlabeled data.                          | `RESULTADOS/.../PSEUDO`      |
| **03**  | `03_train_student.ipynb`     | Trains the Student model on Original + Pseudo-labeled data.          | **Final Student Model**      |
| **04**  | `04_final_eval.py`           | **Official Evaluation**. Tests models against the isolated Test set. | Metrics & Confusion Matrices |
| **05**  | `05_benchmarking.py`         | Measures computational resources (FPS, Parameters).                  | Efficiency Plots             |
| **---** | **Baselines & Ablation**     | *Supplementary experiments*                                          |                              |
| **06**  | `06_baseline_prep.ipynb`     | Prepares flattened data for SVM/RF.                                  | `experimento_baselines/`     |
| **07**  | `07_baselines.ipynb`         | Trains/Evaluates SVM and Random Forest.                              | Baseline Metrics Table       |
| **08**  | `08_gen_ablation_data.ipynb` | Generates datasets for sensitivity analysis.                         | `EXPERIMENTOS_ABLACION/`     |
| **09**  | `09_run_ablation.ipynb`      | Runs training for ablation studies (thresholds, etc.).               | Sensitivity Results          |
| **10**  | `10_data_stats.ipynb`        | Audits class distribution per folder.                                | Distribution Report          |

---

## 🚀 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jzegarragu/LTE-Interference-SemiSupervised.git
   cd LTE-Interference-SemiSupervised
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🔗 Citation

If you use this code or our findings for your research, please cite our work:

```bibtex
@article{zegarra2024lte,
  title={LTE Interference Classification via Semi-Supervised Teacher–Student Approach with CNN and ResNet},
  author={Zegarra Guardamino, Jorge Luis and Maldonado-Quispe, Percy},
  journal={School of Computer Science, UNSA},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
```
