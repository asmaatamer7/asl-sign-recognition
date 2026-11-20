
# 🤟 Isolated Sign Language Recognition
### Pose-Based Deep Learning on ASL Citizen

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

The goal is to classify short sign-language video sequences into their corresponding signs using deep learning models.

Instead of processing raw video frames directly, the project uses pose-based representations extracted with **MediaPipe Holistic**. These representations are then processed by sequence-learning models implemented using **TensorFlow / Keras**.

### Project Information

| Item | Description |
|---|---|
| Course | Deep Learning (AI403) |
| University | Pharos University in Alexandria |
| Selected Idea | #15 — Isolated Sign Language Recognition |
| Dataset | ASL Citizen |
| Task | Multi-class classification |
| Number of Classes | 100 signs |
| Input Representation | Pose sequences |
| Feature Extraction | MediaPipe Holistic |
| Deep Learning Framework | TensorFlow / Keras |

---

## 🏆 Model Performance

Final model comparison and performance results will be added after completing all experiments and evaluating the models using three random seeds.

| Metric | Result |
|---|---|
| Best Model | Pending |
| Test Top-1 Accuracy | Pending |
| Test Top-5 Accuracy | Pending |
| Test Macro F1-score | Pending |
| Number of Seeds | 3 |

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
├── results/
│   └── main/
│       └── baseline/
│           └── seed0/
│               └── metrics.json
│
├── src/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Additional model results and files will be added as experiments are completed.*

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

## 🚀 Training

Training commands and model configurations will be documented here as the final experiment setup is confirmed.

The project uses TensorFlow / Keras for deep learning model implementation.

---

## 📈 Evaluation

Models are evaluated using classification metrics, including:

- Top-1 Accuracy
- Top-5 Accuracy
- Macro F1-score
- Confusion Matrix

Final evaluation results will be added after completing the experiments.

---

## 🖥️ Demo

A demonstration interface is included in the project repository.

**Demo instructions:** Pending.

---

## 💻 Hardware and Computational Cost

| Item | Details |
|---|---|
| Training Hardware | Pending |
| GPU | Pending |
| Training Time | Pending |
| Inference Time | Pending |

---

## 📚 References and Acknowledgments

This project builds upon the following resources and technologies:

- **ASL Citizen** — American Sign Language video dataset.
- **MediaPipe Holistic** — Pose and hand landmark extraction.
- **TensorFlow / Keras** — Deep learning framework.
- **Scikit-learn** — Machine learning utilities and evaluation.

Custom model components and implementation details will be documented in the relevant source files.

---

## 📌 Project Status

<sub>This README will be updated as implementation, experimentation, and evaluation progress.</sub>

**Current Status:** In Progress
