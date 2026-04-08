"""Unit tests for the FedTumorNet data pipeline."""
import os
import sys
import numpy as np
import pytest
from PIL import Image
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.partition import dirichlet_partition, iid_partition, pathological_partition
from src.data.preprocessing import get_train_transforms, get_eval_transforms
from src.data.dataset import BrainTumorDataset, CLASS_MAP

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_labels():
    """Balanced 400-sample label array across 4 classes."""
    return [c for c in range(4) for _ in range(100)]  # 100 per class

@pytest.fixture
def dummy_dataset(tmp_path):
    """Create dummy image files and BrainTumorDataset."""
    paths, labels = [], []
    for cls, idx in CLASS_MAP.items():
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(10):
            img = Image.new("RGB", (128, 128), color=(idx * 60, i * 20, 50))
            p = str(cls_dir / f"{i}.jpg")
            img.save(p)
            paths.append(p)
            labels.append(idx)
    return paths, labels

# ─── Partition Tests ──────────────────────────────────────────────────────────

def test_dirichlet_partition_correct_total(dummy_labels):
    """Total indices across all clients equals total samples."""
    partitions = dirichlet_partition(dummy_labels, num_clients=3, alpha=0.5, seed=42)
    total = sum(len(p) for p in partitions)
    assert total == len(dummy_labels), f"Expected {len(dummy_labels)}, got {total}"

def test_dirichlet_partition_no_overlap(dummy_labels):
    """No sample appears in multiple clients."""
    partitions = dirichlet_partition(dummy_labels, num_clients=3, alpha=0.5, seed=42)
    all_idx = [idx for p in partitions for idx in p]
    assert len(all_idx) == len(set(all_idx)), "Duplicate indices found across clients"

def test_dirichlet_noniid_skew(dummy_labels):
    """With alpha=0.1, at least one client has >50% of one class."""
    labels = np.array(dummy_labels)
    partitions = dirichlet_partition(dummy_labels, num_clients=3, alpha=0.1, seed=42)
    found_skew = False
    for idxs in partitions:
        if len(idxs) == 0: continue
        client_labels = labels[idxs]
        total = len(client_labels)
        for c in range(4):
            if np.sum(client_labels == c) / total > 0.5:
                found_skew = True
                break
    assert found_skew, "Expected non-IID skew with alpha=0.1 but none found"

def test_iid_partition_balance(dummy_labels):
    """IID partition: each client gets ≈equal samples (within 10%)."""
    partitions = iid_partition(dummy_labels, num_clients=3, seed=42)
    sizes = [len(p) for p in partitions]
    mean_size = np.mean(sizes)
    for sz in sizes:
        assert abs(sz - mean_size) / mean_size <= 0.15, f"Size {sz} too far from mean {mean_size:.0f}"

def test_dirichlet_all_clients_nonempty(dummy_labels):
    """All clients receive at least min_samples samples."""
    partitions = dirichlet_partition(dummy_labels, num_clients=3, alpha=0.1,
                                      seed=42, min_samples=5)
    for i, p in enumerate(partitions):
        assert len(p) >= 5, f"Client {i} has only {len(p)} samples"

# ─── Transform Tests ──────────────────────────────────────────────────────────

def test_train_transforms_output_shape():
    """Train transforms produce tensor of shape (3, 224, 224)."""
    tfm = get_train_transforms(224)
    img = Image.new("RGB", (300, 250))
    t = tfm(img)
    assert t.shape == (3, 224, 224), f"Bad shape: {t.shape}"

def test_eval_transforms_output_shape():
    """Eval transforms produce tensor of shape (3, 224, 224)."""
    tfm = get_eval_transforms(224)
    img = Image.new("RGB", (100, 100))
    t = tfm(img)
    assert t.shape == (3, 224, 224), f"Bad shape: {t.shape}"

# ─── Dataset Tests ────────────────────────────────────────────────────────────

def test_dataset_len(dummy_dataset):
    """BrainTumorDataset reports correct length."""
    paths, labels = dummy_dataset
    ds = BrainTumorDataset(paths, labels)
    assert len(ds) == len(paths)

def test_dataset_label_range(dummy_dataset):
    """All labels must be in {0, 1, 2, 3}."""
    paths, labels = dummy_dataset
    tfm = get_eval_transforms(224)
    ds = BrainTumorDataset(paths, labels, transform=tfm)
    for _, lbl in ds:
        assert 0 <= lbl <= 3, f"Label {lbl} out of range"

def test_dataset_returns_tensor(dummy_dataset):
    """Dataset __getitem__ returns (tensor, int) with correct shape."""
    import torch
    paths, labels = dummy_dataset
    tfm = get_eval_transforms(224)
    ds = BrainTumorDataset(paths, labels, transform=tfm)
    img, lbl = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(lbl, int)
