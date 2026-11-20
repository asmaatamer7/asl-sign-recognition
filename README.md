# Isolated Sign Language Recognition

**Pose-Based Machine Learning and Deep Learning on ASL Citizen**

Deep Learning — AI403 | Pharos University in Alexandria  
Academic Year: 2026–2027

## 1. Project Overview

This project focuses on isolated American Sign Language (ASL) sign recognition using pose-based machine learning and deep learning approaches.

The system extracts human body and hand keypoints from video sequences using MediaPipe Holistic, then uses trained models to classify each video into one of 100 sign classes.

The evaluated models are:
- Machine Learning Baseline
- GRU
- Transformer
- ST-GCN

## 2. Dataset

The project uses a 100-class subset of the ASL Citizen dataset.

Raw dataset files are not included in this repository. Dataset preparation and keypoint preprocessing are handled separately.

## 3. Experimental Results

Mean Top-1 Accuracy across three seeds:

| Model | Top-1 Accuracy |
|---|---:|
| Transformer | 80.0% |
| GRU | 76.6% |
| Baseline | 56.8% |
| ST-GCN | 52.0% |

The Transformer achieved the best Top-1 accuracy, while GRU achieved the best Top-5 accuracy (92.5%).

## 4. Repository Structure

```text
configs/       Experiment configurations
data/          Dataset documentation and preprocessing
src/           Dataset loaders, models, training and evaluation
experiments/   Experiment outputs and results
demo/          Gradio application
report/        Technical report and presentation
contributions/ Team contributions
```

## 5. Environment Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

The demo was tested with Python 3.12, TensorFlow 2.19.1, MediaPipe 0.10.21, and Gradio.

## 6. Training

Training configurations are available in the `configs/` directory.

Training scripts are located in `src/`.

Run the corresponding training script and configuration for each model.

## 7. Evaluation

Evaluation and experiment results are stored under `experiments/`.

Aggregated comparison tables and figures are available in the project aggregation outputs.

## 8. Demo

The project includes an interactive Gradio demo supporting video upload and webcam recording.

Run the demo with trained checkpoints:

```bash
python demo/app.py --run \
experiments/main/gru/seed0 \
experiments/main/STGCN/seed0 \
experiments/main/Transformer/seed0 \
--share
```

The demo extracts keypoints from a video and displays the Top-5 predicted signs.

**Note:** The demo recognizes isolated signs from the 100-class vocabulary. It does not perform continuous sentence translation.

## 9. Hardware and Runtime

Experiments were conducted using Google Colab GPU environments.

Training time and inference performance vary depending on the model and hardware. Detailed measurements are included in the experiment results.

## 10. Credits

- ASL Citizen dataset.
- MediaPipe for pose and hand landmark extraction.
- TensorFlow and related machine learning libraries.
- Gradio for the interactive demo.
## Ablation Study

Ablation experiments were defined to investigate the contribution of
attention mechanisms, joint selection, and input representation.

The planned experiments include:
- Transformer without attention.
- Transformer using hand joints only.
- Transformer using body joints only.
- ST-GCN using hand joints only.
- ST-GCN using body joints only.

These experiments were not completed within the project timeline.
Therefore, no ablation results are reported.
## Final Model Results

All four model families were evaluated using three random seeds.

| Model | Top-1 Accuracy | Top-5 Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Transformer | 80.0% ± 1.3 | 92.4% ± 0.3 | 79.7% ± 1.4 |
| GRU | 76.6% ± 0.8 | 92.5% ± 0.2 | 76.5% ± 0.9 |
| Baseline | 56.8% ± 0.0 | 80.2% ± 0.0 | 57.0% ± 0.0 |
| ST-GCN | 52.0% ± 1.3 | 80.4% ± 0.3 | 50.9% ± 1.2 |

The Transformer achieved the highest Top-1 accuracy and Macro-F1,
while GRU achieved the highest Top-5 accuracy.
