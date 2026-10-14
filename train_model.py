"""
Training script for SignBridge AI HandLandmarkMLP model.
Loads dataset, normalizes landmarks, performs stratified train/val/test split,
trains PyTorch MLP, evaluates on held-out test set, and saves model artifacts.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "ml" / "scripts"))

from app.models.neural_net import HandLandmarkMLP
from landmark_normalization import normalize_landmarks

def train_model(
    data_path: str = None,
    artifacts_dir: str = None,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
):
    print("=" * 60)
    print("SignBridge AI — Model Training Pipeline (PyTorch HandLandmarkMLP)")
    print("=" * 60)

    if data_path is None:
        data_path = PROJECT_ROOT / "ml" / "data" / "dataset.json"
    else:
        data_path = Path(data_path)

    if artifacts_dir is None:
        artifacts_dir = PROJECT_ROOT / "ml" / "artifacts"
    else:
        artifacts_dir = Path(artifacts_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_baseline_dataset.py or collect data first.")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} samples from {data_path}")

    # 1. Normalize and extract features
    X_raw = []
    y_raw = []
    for item in raw_data:
        pts = item["landmarks"]
        handedness = item.get("handedness", "Right")
        norm_pts = normalize_landmarks(pts, handedness=handedness)
        X_raw.append(norm_pts)
        y_raw.append(item["label"].upper())

    unique_classes = sorted(list(set(y_raw)))
    num_classes = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for i, c in enumerate(unique_classes)}

    print(f"Target classes ({num_classes}): {unique_classes}")

    X = np.array(X_raw, dtype=np.float32)
    y = np.array([class_to_idx[lbl] for lbl in y_raw], dtype=np.int64)

    # 2. Stratified train/val/test split
    total_holdout = val_split + test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=total_holdout, random_state=random_seed, stratify=y
    )
    rel_test = test_split / total_holdout
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=random_seed, stratify=y_temp
    )

    print(f"Split sizes — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size, shuffle=False
    )

    # 3. Model & Optimizer
    model = HandLandmarkMLP(input_dim=63, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    start_time = time.time()
    best_val_loss = float("inf")
    best_weights = None
    history = []

    print("\n--- Training Started ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = torch.argmax(out, dim=1)
            train_correct += (preds == by).sum().item()
            train_total += bx.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += bx.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4)
        })

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")

    # 4. Evaluation on Held-out Test Set
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(by.cpu().numpy().tolist())

    test_acc = accuracy_score(all_targets, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes))).tolist()
    rep = classification_report(
        all_targets, all_preds, labels=list(range(num_classes)),
        target_names=unique_classes, output_dict=True, zero_division=0
    )

    print("\n--- Test Set Evaluation ---")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall:    {rec_macro:.4f}")
    print(f"Macro F1-Score:  {f1_macro:.4f}")

    per_class = {}
    for c in unique_classes:
        c_rep = rep.get(c, {})
        per_class[c] = {
            "precision": round(float(c_rep.get("precision", 0.0)), 4),
            "recall": round(float(c_rep.get("recall", 0.0)), 4),
            "f1_score": round(float(c_rep.get("f1-score", 0.0)), 4),
            "support": int(c_rep.get("support", 0))
        }

    # 5. Save Artifacts
    model_save_path = artifacts_dir / "best_model.pth"
    classes_save_path = artifacts_dir / "classes.json"
    eval_save_path = artifacts_dir / "evaluation_report.json"
    meta_save_path = artifacts_dir / "model_meta.json"

    torch.save({
        "state_dict": model.state_dict(),
        "metadata": {
            "classes": unique_classes,
            "test_accuracy": float(test_acc),
            "timestamp": datetime.utcnow().isoformat()
        }
    }, model_save_path)

    with open(classes_save_path, "w", encoding="utf-8") as f:
        json.dump(unique_classes, f, indent=2)

    eval_data = {
        "model_version": f"mlp_{int(time.time())}",
        "model_type": "mlp",
        "num_classes": num_classes,
        "test_accuracy": round(float(test_acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "labels": unique_classes,
        "training_duration_seconds": round(duration, 2),
        "evaluated_at": datetime.utcnow().isoformat(),
        "dataset_size": len(raw_data),
        "is_trained": True
    }
    with open(eval_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": eval_data["model_version"],
            "model_type": "mlp",
            "trained_at": eval_data["evaluated_at"],
            "test_accuracy": round(float(test_acc), 4),
            "classes_count": num_classes
        }, f, indent=2)

    print(f"\nSaved model weights to: {model_save_path}")
    print(f"Saved evaluation report to: {eval_save_path}")
    return eval_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
