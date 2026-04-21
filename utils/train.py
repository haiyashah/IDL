"""
Training Loop
=============
Unified training script for LSTM baseline and TFT variant.
Logs to Weights & Biases and saves the best checkpoint by val RMSE.

Usage
-----
python train.py --model lstm --subset FD001 --epochs 50
python train.py --model tft  --subset FD001 --epochs 50
python train.py --model lstm --subset FD003 --epochs 50
python train.py --model tft  --subset FD003 --epochs 50
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warn] wandb not installed — logging to console only.")

from utils.data import load_cmapss
from utils.metrics import evaluate
from models.lstm_baseline import LSTMBaseline
from models.tft import TemporalFusionTransformer


# ------------------------------------------------------------------
# Defaults (tune these via CLI args or W&B sweeps)
# ------------------------------------------------------------------
DEFAULTS = {
    "lstm": dict(hidden_size=128, num_layers=2, dropout=0.2, lr=1e-3, batch_size=64),
    "tft":  dict(d_model=64, n_heads=4, n_lstm_layers=2, dropout=0.1, lr=5e-4, batch_size=64),
}


def build_model(model_name: str, n_features: int, hparams: dict) -> nn.Module:
    if model_name == "lstm":
        return LSTMBaseline(
            n_features=n_features,
            hidden_size=hparams.get("hidden_size", 128),
            num_layers=hparams.get("num_layers", 2),
            dropout=hparams.get("dropout", 0.2),
        )
    elif model_name == "tft":
        return TemporalFusionTransformer(
            n_features=n_features,
            d_model=hparams.get("d_model", 64),
            n_heads=hparams.get("n_heads", 4),
            n_lstm_layers=hparams.get("n_lstm_layers", 2),
            dropout=hparams.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x).squeeze(-1)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hparams = DEFAULTS[args.model].copy()
    hparams.update({"subset": args.subset, "seq_len": args.seq_len, "epochs": args.epochs})

    # W&B init
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project="cmapss-rul",
            name=f"{args.model}_{args.subset}",
            config=hparams,
        )

    # Data
    train_loader, val_loader, test_loader, n_features = load_cmapss(
        data_dir=args.data_dir,
        subset=args.subset,
        seq_len=args.seq_len,
        batch_size=hparams["batch_size"],
    )

    # Model
    model = build_model(args.model, n_features, hparams).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}  Params: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=hparams["lr"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.subset}_best.pt")

    best_val_rmse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_rmse, val_phm, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_rmse={val_rmse:.4f}  "
            f"val_phm={val_phm:.2f}  "
            f"lr={lr_now:.2e}"
        )

        if WANDB_AVAILABLE and not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "val_phm": val_phm,
                "lr": lr_now,
            })

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_rmse": val_rmse, "val_phm": val_phm})

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_rmse": val_rmse,
                "hparams": hparams,
                "model_name": args.model,
                "subset": args.subset,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint (val_rmse={val_rmse:.4f})")

    # Final test evaluation with best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_rmse, test_phm, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\n=== Test Results ({args.model.upper()} / {args.subset}) ===")
    print(f"  RMSE : {test_rmse:.4f}")
    print(f"  PHM  : {test_phm:.2f}")

    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.log({"test_rmse": test_rmse, "test_phm": test_phm})
        wandb.finish()

    # Save results JSON for reproduce notebook
    os.makedirs(args.results_dir, exist_ok=True)
    results = {
        "model": args.model,
        "subset": args.subset,
        "test_rmse": test_rmse,
        "test_phm": test_phm,
        "best_val_rmse": best_val_rmse,
        "history": history,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    results_path = os.path.join(args.results_dir, f"{args.model}_{args.subset}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return test_rmse, test_phm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "tft"], default="lstm")
    p.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    p.add_argument("--data_dir", default="./data/CMAPSSData")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
