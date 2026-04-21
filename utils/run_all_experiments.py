#!/usr/bin/env python3
"""
run_all_experiments.py
======================
Trains all four combinations (LSTM/TFT × FD001/FD003) sequentially
and prints a final summary table.

Usage
-----
python run_all_experiments.py --data_dir ./data/CMAPSSData [--epochs 50] [--no_wandb]
"""

import argparse
import subprocess
import sys
import json
import os


EXPERIMENTS = [
    ("lstm", "FD001"),
    ("tft",  "FD001"),
    ("lstm", "FD003"),
    ("tft",  "FD003"),
]


def run_experiment(model, subset, args):
    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--subset", subset,
        "--data_dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--seq_len", str(args.seq_len),
        "--checkpoint_dir", args.checkpoint_dir,
        "--results_dir", args.results_dir,
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")

    print(f"\n{'='*60}")
    print(f"Running: {model.upper()} / {subset}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def print_summary(results_dir):
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Subset':<8} {'RMSE':>10} {'PHM':>12}")
    print("-" * 60)
    for model, subset in EXPERIMENTS:
        path = os.path.join(results_dir, f"{model}_{subset}.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            print(f"{model.upper():<10} {subset:<8} {r['test_rmse']:>10.4f} {r['test_phm']:>12.2f}")
        else:
            print(f"{model.upper():<10} {subset:<8} {'N/A':>10} {'N/A':>12}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/CMAPSSData")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    for model, subset in EXPERIMENTS:
        success = run_experiment(model, subset, args)
        if not success:
            print(f"[ERROR] {model}/{subset} failed — stopping.")
            sys.exit(1)

    print_summary(args.results_dir)


if __name__ == "__main__":
    main()
