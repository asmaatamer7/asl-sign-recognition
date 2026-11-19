#!/bin/bash
# end-to-end test on synthetic data (a few minutes on CPU)
set -e
python tests/make_fake_data.py
python -m src.tune --model gru --trials 2 --epochs 3
python -m src.run_all --study main aug ablation repr --seeds 0 1 --epochs 3 --only gru transformer stgcn baseline
python -m src.pretrain --config configs/transformer.yaml --epochs 2
python -m src.run_all --study pretrain --seeds 0 1 --epochs 3
python -m src.error_analysis --model stgcn
python -m src.aggregate
