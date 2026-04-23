"""PyTorch Dataset and DataLoader utilities for Brain Tumor MRI."""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import get_train_transforms, get_eval_transforms
from src.data.partition import dirichlet_partition, visualize_partition

CLASS_MAP = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
CLASS_NAMES = list(CLASS_MAP.keys())
NUM_CLASSES = 4

class BrainTumorDataset(Dataset):
    """Brain Tumor MRI Dataset — wraps image paths + labels."""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

def load_image_paths_and_labels(data_dir: str) -> Tuple[List[str], List[int]]:
    """Scan directory for images; assumes structure: data_dir/{class_name}/*.jpg"""
    paths, labels = [], []
    data_dir = Path(data_dir)
    for cls, lbl in CLASS_MAP.items():
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            for d in data_dir.iterdir():
                if d.name.lower() == cls.lower() and d.is_dir(): cls_dir = d; break
        if cls_dir.exists():
            for ext in ["*.jpg","*.jpeg","*.png"]:
                for p in sorted(cls_dir.glob(ext)):
                    paths.append(str(p)); labels.append(lbl)
    return paths, labels

def get_centralized_dataloaders(data_dir="data/raw", batch_size=32, val_ratio=0.15,
                                test_ratio=0.15, image_size=224, num_workers=4, seed=42):
    train_dir = Path(data_dir)/"Training"; test_dir = Path(data_dir)/"Testing"
    tr_tfm = get_train_transforms(image_size); ev_tfm = get_eval_transforms(image_size)
    if train_dir.exists() and test_dir.exists():
        tp, tl = load_image_paths_and_labels(train_dir)
        xp, xl = load_image_paths_and_labels(test_dir)
        tr_idx, v_idx = train_test_split(range(len(tp)), test_size=val_ratio,
                                          stratify=tl, random_state=seed)
        vp = [tp[i] for i in v_idx]; vl = [tl[i] for i in v_idx]
        ftp = [tp[i] for i in tr_idx]; ftl = [tl[i] for i in tr_idx]
        xp2, xl2 = xp, xl
    else:
        ap, al = load_image_paths_and_labels(data_dir)
        tri, tei = train_test_split(range(len(ap)), test_size=test_ratio, stratify=al, random_state=seed)
        rp = [ap[i] for i in tri]; rl = [al[i] for i in tri]
        tri2, vi = train_test_split(range(len(rp)), test_size=val_ratio/(1-test_ratio),
                                     stratify=rl, random_state=seed)
        ftp=[rp[i] for i in tri2]; ftl=[rl[i] for i in tri2]
        vp=[rp[i] for i in vi]; vl=[rl[i] for i in vi]
        xp2=[ap[i] for i in tei]; xl2=[al[i] for i in tei]
    ds_tr = BrainTumorDataset(ftp, ftl, tr_tfm)
    ds_v  = BrainTumorDataset(vp, vl, ev_tfm)
    ds_te = BrainTumorDataset(xp2, xl2, ev_tfm)
    print(f"Centralized: train={len(ds_tr)}, val={len(ds_v)}, test={len(ds_te)}")
    mk = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return {"train": DataLoader(ds_tr, batch_size, shuffle=True, drop_last=True, **mk),
            "val":   DataLoader(ds_v,  batch_size, shuffle=False, **mk),
            "test":  DataLoader(ds_te, batch_size, shuffle=False, **mk)}

def _safe_split(indices, labels, test_size, seed):
    """train_test_split with stratify, falling back to non-stratified if any
    class has < 2 members (common with small non-IID client shards)."""
    from collections import Counter
    counts = Counter(labels)
    can_stratify = len(indices) >= 4 and all(c >= 2 for c in counts.values())
    try:
        return train_test_split(
            indices, test_size=test_size,
            stratify=labels if can_stratify else None,
            random_state=seed,
        )
    except ValueError:
        # Last-resort fallback: never let a small shard crash training
        return train_test_split(indices, test_size=test_size,
                                stratify=None, random_state=seed)


def create_federated_datasets(data_dir="data/raw/Training", num_clients=3, alpha=0.5,
                               val_ratio=0.15, test_ratio=0.15, image_size=224, seed=42):
    ap, al = load_image_paths_and_labels(data_dir)
    partitions = dirichlet_partition(al, num_clients, alpha, seed=seed)
    tr_tfm = get_train_transforms(image_size); ev_tfm = get_eval_transforms(image_size)
    client_datasets = {}
    for cid, idxs in enumerate(partitions):
        cp = [ap[i] for i in idxs]
        cl = [al[i] for i in idxs]

        # Split off val+test pool from training shard
        ti, tmpi = _safe_split(list(range(len(cp))), cl,
                                test_size=val_ratio + test_ratio, seed=seed + cid)
        tep = [cp[i] for i in tmpi]
        tel = [cl[i] for i in tmpi]

        # Split val+test pool into val / test
        rel_t = test_ratio / (val_ratio + test_ratio)
        if len(tep) < 2:
            # Pool too small — use everything as val, empty test
            vi, tei = list(range(len(tep))), []
        else:
            vi, tei = _safe_split(list(range(len(tep))), tel,
                                   test_size=rel_t, seed=seed + cid)

        client_datasets[cid] = {
            "train": BrainTumorDataset([cp[i] for i in ti],  [cl[i] for i in ti],  tr_tfm),
            "val":   BrainTumorDataset([tep[i] for i in vi], [tel[i] for i in vi], ev_tfm),
            "test":  BrainTumorDataset([tep[i] for i in tei],[tel[i] for i in tei],ev_tfm),
        }

    test_dir = Path(data_dir).parent / "Testing"
    if test_dir.exists():
        gp, gl = load_image_paths_and_labels(test_dir)
    else:
        _, tei2 = _safe_split(list(range(len(ap))), al, test_size=test_ratio, seed=seed)
        gp = [ap[i] for i in tei2]
        gl = [al[i] for i in tei2]

    global_test = BrainTumorDataset(gp, gl, ev_tfm)
    print(f"\nFederated: {num_clients} clients, alpha={alpha}")
    for cid, ds in client_datasets.items():
        print(f"  Client {cid}: train={len(ds['train'])}, "
              f"val={len(ds['val'])}, test={len(ds['test'])}")
    print(f"  Global test: {len(global_test)}")
    visualize_partition(partitions, al, CLASS_NAMES,
        save_path=f"outputs/figures/eda/partition_alpha_{alpha}.png",
        title=f"Non-IID Partition (α={alpha})")
    return client_datasets, global_test


def get_dataloaders(client_datasets, batch_size=32, num_workers=4):
    # Only pin memory when a CUDA GPU is actually available
    import torch as _torch
    mk = dict(num_workers=num_workers, pin_memory=_torch.cuda.is_available())
    return {cid: {
        "train": DataLoader(ds["train"], batch_size, shuffle=True, drop_last=True, **mk),
        "val":   DataLoader(ds["val"],   batch_size, shuffle=False, **mk),
        "test":  DataLoader(ds["test"],  batch_size, shuffle=False, **mk),
    } for cid, ds in client_datasets.items()}
