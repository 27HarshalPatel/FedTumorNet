"""Training loop for centralized baseline and shared FL client logic."""
import copy
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.utils.metrics import MetricsTracker, compute_accuracy, compute_confusion_matrix
from src.utils.metrics import compute_auc_roc, compute_f1, compute_classification_report

def get_device(device_str="auto"):
    if device_str == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)

def train_one_epoch(model, dataloader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = compute_accuracy(all_preds, all_labels)
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device) -> Tuple[float, float, list, list, list]:
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = compute_accuracy(all_preds, all_labels)
    return avg_loss, acc, all_preds, all_labels, all_probs

def train_centralized(config: Dict):
    """Full centralized training loop with early stopping."""
    from src.data.dataset import get_centralized_dataloaders
    from src.models.resnet import get_model
    import yaml

    seed = config["training"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    device = get_device(config["training"]["device"])
    print(f"Device: {device}")

    loaders = get_centralized_dataloaders(
        data_dir="data/raw",
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", None),
        seed=seed,
    )

    model = get_model(config["training"]["model"], config["training"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=config["training"]["learning_rate"],
                     weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])

    tracker = MetricsTracker()
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        va_loss, va_acc, _, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()
        tracker.record("train", tr_loss, tr_acc)
        tracker.record("val", va_loss, va_acc)
        print(f"Epoch {epoch:3d} | Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
              f"Val Loss={va_loss:.4f} Acc={va_acc:.4f}")
        # Early stopping
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_dir / "centralized_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best model and evaluate on test
    model.load_state_dict(best_model_state)
    te_loss, te_acc, preds, labels, probs = evaluate(model, loaders["test"], criterion, device)
    print(f"\n{'='*60}\nCentralized Test Results\n{'='*60}")
    print(f"Test Loss: {te_loss:.4f} | Test Accuracy: {te_acc:.4f}")

    auc = compute_auc_roc(np.array(probs), labels)
    f1  = compute_f1(preds, labels)
    report = compute_classification_report(preds, labels)
    print(f"Macro AUC: {auc['macro']:.4f} | Macro F1: {f1['macro']:.4f}")

    compute_confusion_matrix(preds, labels,
        save_path="outputs/figures/centralized_confusion_matrix.png")
    tracker.plot_training_curves("outputs/figures/centralized_training_curves.png")
    tracker.save_to_json("outputs/checkpoints/centralized_metrics.json")

    return model, tracker, {"test_acc": te_acc, "auc": auc, "f1": f1, "report": report}
