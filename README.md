# 🤟 Isolated Sign Language Recognition
### Pose-Based Machine Learning and Deep Learning on ASL Citizen

**Deep Learning — AI403 | Pharos University in Alexandria**  
**Academic Year: 2026–2027 | Fall 2026**

---

## 👥 Team Members

| # | Student Name | Student ID |
|---|---|---|
| 1 | Asmaa Tamer Farahat Saad | 202302724 |
| 2 | Hana Emad El Din Shazly | 202302840 |
| 3 | Maram Mohamed Ashour | 202302848 |
| 4 | Noureen Mohamed Ashour | 202302849 |

---

## 🎯 Project Overview

This project focuses on **Isolated American Sign Language (ASL) Recognition**.

The goal is to classify short sign-language video sequences into their corresponding signs using machine learning and deep learning models.

Instead of processing raw video frames directly, the project uses pose-based representations extracted using **MediaPipe Holistic**. The extracted keypoints are processed to create structured temporal features that can be used by different classification architectures.

The project explores a classical machine learning baseline and multiple deep learning approaches for sequence classification.

### Project Information

| Item | Description |
|---|---|
| Course | Deep Learning (AI403) |
| University | Pharos University in Alexandria |
| Selected Idea | #15 — Isolated Sign Language Recognition |
| Dataset | ASL Citizen |
| Task | Multi-class classification |
| Number of Classes | 100 signs |
| Input Representation | Pose and hand keypoint sequences |
| Feature Extraction | MediaPipe Holistic |
| Classical ML | Logistic Regression |
| Deep Learning Framework | TensorFlow / Keras and other model-specific implementations |

---

## 🧠 Models

The project investigates the following models:

| Model | Description | Status |
|---|---|---|
| Baseline | Logistic Regression using handcrafted temporal features | Completed |
| GRU | Gated Recurrent Unit for sequential feature learning | Experiment completed / results pending final verification |
| Transformer | Attention-based sequence classification | Pending final results |
| ST-GCN | Spatial-Temporal Graph Convolutional Network | Pending final results |

Final model comparisons will be updated after all experiments have been verified.

---

## 📊 Baseline Experimental Results

The baseline uses **Logistic Regression** with handcrafted temporal features extracted from pose sequences.

### Hyperparameter Tuning

The regularization parameter `C` was selected using validation performance.

| Parameter | Value |
|---|---|
| Model | Logistic Regression |
| Hyperparameter | C |
| Search Values | 0.001, 0.01, 0.1, 1.0, 10.0, 100.0 |
| Number of Trials | 6 |
| Selection Metric | Validation Top-1 Accuracy |
| Best C | 1.0 |

### Final Evaluation

The selected configuration was evaluated using three seeds: 0, 1, and 2.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Validation Top-1 Accuracy | 71.35% | 0.00% |
| Validation Top-5 Accuracy | 89.06% | 0.00% |
| Validation Macro F1-score | 71.37% | 0.00% |
| Test Top-1 Accuracy | 56.79% | 0.00% |
| Test Top-5 Accuracy | 80.16% | 0.00% |
| Test Macro F1-score | 56.99% | 0.00% |

**Note:** The baseline produces identical predictions across the three seeds because its handcrafted feature extraction and Logistic Regression training procedure are deterministic under the selected solver configuration.

### Computational Cost

| Metric | Result |
|---|---:|
| Model Parameters / Features | 68,700 |
| Model Size | 0.57 MB |
| Training Time | Approximately 10–14 seconds |
| Hardware | CPU (x86_64) |

---

## 📂 Repository Structure

```text
asl-sign-recognition/
│
├── configs/
│   └── baseline.yaml
│
├── data/
│   └── DATASET.md
│
├── demo/
│
├── experiments/
│   ├── tuning/
│   └── main/
│       └── baseline/
│           ├── seed0/
│           ├── seed1/
│           └── seed2/
│
├── src/
│
├── requirements.txt
├── README.md
└── .gitignore
```

Additional configuration files, model implementations, and experiment outputs may be added as the project develops.

Large model artifacts and datasets are managed separately and are not necessarily included in the GitHub repository.

---

## 📊 Dataset

The project uses a 100-class subset of the **ASL Citizen** dataset.

The dataset contains isolated sign-language videos performed by different signers.

### Dataset Preparation

1. Select a subset of 100 signs.
2. Use the official signer-independent train, validation, and test splits.
3. Extract pose and hand keypoints using MediaPipe Holistic.
4. Normalize the extracted keypoints.
5. Prepare fixed-length sequences for model training.

### Dataset Split

| Split | Number of Samples |
|---|---:|
| Training | 1,490 |
| Validation | 384 |
| Testing | 1,215 |
| **Total** | **3,089** |

**Note:** Raw videos and extracted keypoint files are not stored in this GitHub repository.

For dataset setup and access instructions, see [`data/DATASET.md`](data/DATASET.md).

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/asmaatamer7/asl-sign-recognition.git
cd asl-sign-recognition
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training and Experiments

The project includes separate implementations and configurations for the evaluated models.

Each model is trained and evaluated using the project's experimental setup.

Hyperparameter tuning is performed using validation performance. The test set is reserved for final evaluation.

Refer to the source code and configuration files for model-specific execution commands.

---

## 📈 Evaluation

Models are evaluated using the following metrics:

- Top-1 Accuracy
- Top-5 Accuracy
- Macro F1-score
- Confusion Matrix

Final model comparisons will use the same evaluation metrics and experimental protocol wherever applicable.

---

## 🖥️ Demo

A demonstration interface is included in the project repository.

**Demo instructions:** To be documented.

---

## 💻 Hardware and Computational Cost

Hardware and computational cost are reported for individual models based on their experimental runs.

| Item | Details |
|---|---|
| Baseline Training Hardware | CPU (x86_64) |
| Baseline Model Size | 0.57 MB |
| Baseline Training Time | Approximately 10–14 seconds |
| Other Models | To be reported after final evaluation |

---

## 📚 References and Acknowledgments

This project builds upon the following resources and technologies:

- **ASL Citizen** — American Sign Language video dataset.
- **MediaPipe Holistic** — Pose and hand landmark extraction.
- **TensorFlow / Keras** — Deep learning framework.
- **Scikit-learn** — Classical machine learning and evaluation utilities.

Custom model components and implementation details are documented in the relevant source files.

---

## 📌 Project Status

**Current Status:** Experiments completed or under final verification; final cross-model comparison and documentation are in progress.

The README will be updated with the verified results of all models before final submission.
