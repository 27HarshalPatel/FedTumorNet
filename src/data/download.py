"""Download the Kaggle Brain Tumor MRI dataset (masoudnickparvar/brain-tumor-mri-dataset)."""
import os, json, shutil
from datetime import datetime
from pathlib import Path
from collections import Counter
import yaml
from PIL import Image

def load_config(p="configs/data_config.yaml"):
    with open(p) as f: return yaml.safe_load(f)

def download_dataset(config=None):
    if config is None: config = load_config()
    raw_dir = Path(config["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_dir = raw_dir / "Training"
    test_dir  = raw_dir / "Testing"
    if train_dir.exists() and test_dir.exists():
        print(f"Dataset already exists at {raw_dir}."); return raw_dir
    print("Downloading Brain Tumor MRI Dataset from Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download(config["dataset"]["kaggle_slug"])
        print(f"Downloaded to: {path}")
        downloaded = Path(path)
        for split in ["Training","Testing"]:
            src = downloaded / split; dst = raw_dir / split
            if not dst.exists():
                if src.exists(): shutil.copytree(str(src), str(dst))
                else:
                    for alt in downloaded.iterdir():
                        alt_src = alt / split
                        if alt_src.exists(): shutil.copytree(str(alt_src), str(dst)); break
    except ImportError:
        os.system(f'kaggle datasets download -d {config["dataset"]["kaggle_slug"]} --unzip -p {raw_dir}')
    return raw_dir

def validate_dataset(raw_dir, expected_classes=None):
    raw_dir = Path(raw_dir)
    if expected_classes is None: expected_classes = ["glioma","meningioma","notumor","pituitary"]
    stats = {"splits": {}, "total_images": 0, "classes": expected_classes}
    for split in ["Training","Testing"]:
        split_dir = raw_dir / split
        if not split_dir.exists(): continue
        split_stats = {}
        for cls in expected_classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                for d in split_dir.iterdir():
                    if d.name.lower() == cls.lower(): cls_dir = d; break
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png")) if cls_dir.exists() else []
            split_stats[cls] = len(imgs); stats["total_images"] += len(imgs)
        stats["splits"][split] = split_stats
    print("\n" + "="*60 + "\nDATASET STATISTICS\n" + "="*60)
    for split, counts in stats["splits"].items():
        total = sum(counts.values()); print(f"\n{split}: {total} images\n" + "-"*40)
        for cls, cnt in counts.items():
            pct = cnt/total*100 if total else 0
            print(f"  {cls:15s} {cnt:5d} ({pct:5.1f}%) " + "█"*int(pct/2))
    print(f"\nTotal: {stats['total_images']}\n" + "="*60)
    return stats

def save_metadata(raw_dir, stats):
    meta = {"download_timestamp": datetime.now().isoformat(),
            "source": "kaggle/masoudnickparvar/brain-tumor-mri-dataset",
            "statistics": stats["splits"], "total_images": stats["total_images"],
            "classes": stats["classes"]}
    p = Path(raw_dir)/"metadata.json"
    with open(p,"w") as f: json.dump(meta, f, indent=2)
    print(f"Metadata saved to {p}")

def main():
    config = load_config()
    raw_dir = download_dataset(config)
    stats = validate_dataset(raw_dir, config["dataset"]["classes"])
    save_metadata(raw_dir, stats)
    print("\n✅ Dataset download and validation complete!")

if __name__ == "__main__": main()
