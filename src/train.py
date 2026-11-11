"""
Training Module for Speech Emotion Recognition (SER).

Trains EmotionCNN1D or EmotionCRNN with:
- Class-weighted CrossEntropyLoss (to account for Neutral having half the sample count).
- AdamW optimizer with CosineAnnealingLR scheduler.
- Strict Speaker-Independent evaluation on validation and test sets.
- Checkpoint saving of the best model based on macro F1-score.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from src.model import get_model
from src.dataset import EMOTION_MAP, IDX_TO_EMOTION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def compute_class_weights(labels: np.ndarray, num_classes: int = 8) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # weights inversely proportional to class frequencies
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluate model on a DataLoader, returning loss, accuracy, and macro F1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, np.array(all_preds), np.array(all_targets)


def train_model(
    model_name: str = "cnn1d",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "auto"
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / "ravdess_features.npz"

    if not features_path.exists():
        raise FileNotFoundError(f"Processed features not found at {features_path}. Run src.preprocess first.")

    data = np.load(features_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Loaded datasets from {features_path}:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples (unseen actors)")
    print(f"  Test:  {X_test.shape[0]} samples (unseen actors)")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on device: {device}")

    # Build Tensor datasets
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Class weights
    class_weights = compute_class_weights(y_train, num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Instantiate model
    model = get_model(model_name=model_name, in_features=X_train.shape[1], num_classes=8).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    best_epoch = 0
    checkpoint_path = MODELS_DIR / f"best_{model_name}.pt"

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            running_loss += loss.item() * len(y_b)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(y_b.cpu().numpy())

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "in_features": X_train.shape[1],
                "num_classes": 8
            }, checkpoint_path)
            mark = " ★ Best"
        else:
            mark = ""

        if epoch % 5 == 0 or mark:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% F1: {val_f1*100:.1f}%{mark}")

    print(f"\nTraining finished! Best model saved from Epoch {best_epoch} (Val F1: {best_val_f1*100:.2f}%) to {checkpoint_path}")

    # Load best checkpoint and evaluate on unseen Test set
    print("\n--- Final Test Set Evaluation (Unseen Actors 21-24) ---")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro F1: {test_f1*100:.2f}%")

    np.savez(
        PROCESSED_DIR / f"test_results_{model_name}.npz",
        preds=test_preds,
        targets=test_targets,
        test_acc=test_acc,
        test_f1=test_f1,
        history=history
    )
    return model, test_acc, test_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Model")
    parser.add_argument("--model", type=str, default="cnn1d", choices=["cnn1d", "crnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
