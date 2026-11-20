"""Live demo: video upload or webcam clip -> MediaPipe keypoints -> model -> top-5 signs.
  python demo/app.py --run experiments/main/stgcn/seed0 [--share]
Several models can be loaded:  --run experiments/main/stgcn/seed0 experiments/main/transformer/seed0"""
import argparse, sys, time
from pathlib import Path
import numpy as np, tensorflow as tf, gradio as gr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import load_run
from src.datasets.dataset import sequence_to_tensor
from src.datasets.keypoints import make_holistic, video_to_keypoints

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs="+", required=True); ap.add_argument("--share", action="store_true")
a = ap.parse_args()
MODELS = {}
for r in a.run:
    ck = load_run(r)
    if "model" in ck: MODELS[Path(r).parent.name] = ck           # neural models only (baseline skipped)
HOL = make_holistic()


def predict(video, model_name):
    if video is None:
        return None, "Please record or upload a video of ONE sign."
    t0 = time.time()
    raw = video_to_keypoints(video, HOL)
    if raw is None:
        return None, "Could not read the video."
    hands = (np.abs(raw[:, 33:75]).sum((1, 2)) > 0).mean()
    ck = MODELS[model_name]
    x = sequence_to_tensor(raw, ck["cfg"]["data"])
    p = tf.nn.softmax(ck["model"](x, training=False))[0].numpy()
    top = np.argsort(-p)[:5]
    msg = f"Hands detected in {hands * 100:.0f}% of frames · total {1000 * (time.time() - t0):.0f} ms"
    if hands < 0.5: msg += " ⚠️ hands poorly tracked: keep both hands visible, good light, plain background"
    return {ck["classes"][i]: float(p[i]) for i in top}, msg


with gr.Blocks(title="Isolated sign recognition (ASL Citizen subset)") as demo:
    gr.Markdown("## Isolated sign recognition\nRecord or upload a short clip of one ASL sign "
                "(vocabulary = the 100-sign subset the models were trained on).")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(sources=["webcam", "upload"], label="Sign video")
            mdl = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Model")
            btn = gr.Button("Recognise", variant="primary")
        with gr.Column():
            lab = gr.Label(num_top_classes=5, label="Top-5 predictions"); info = gr.Markdown()
    btn.click(predict, [vid, mdl], [lab, info])

demo.launch(share=a.share)
