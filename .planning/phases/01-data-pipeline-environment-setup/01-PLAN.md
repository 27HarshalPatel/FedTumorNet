---
phase: 1
plan: 01
title: "Data Pipeline & Environment Setup"
wave: 1
depends_on: []
files_modified:
  - requirements.txt
  - setup.py
  - src/data/download.py
  - src/data/preprocessing.py
  - src/data/partition.py
  - src/data/dataset.py
  - configs/data_config.yaml
  - notebooks/01_eda.ipynb
  - tests/test_data_pipeline.py
  - README.md
  - .gitignore
requirements_addressed: [REQ-001]
autonomous: true
---

# Phase 1: Data Pipeline & Environment Setup

<objective>
Set up the complete development environment, download the Kaggle Brain Tumor MRI Dataset (7,023 images, 4 classes: glioma, meningioma, pituitary, no_tumor), implement data preprocessing, non-IID partitioning for federated learning simulation, and create an EDA notebook.
</objective>

## Tasks

<task id="1.1" title="Project Structure & Dependencies">
<read_first>
- README.md (if exists)
</read_first>

<action>
Create the following project directory structure:

```
Medical/
├── configs/
│   └── data_config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── preprocessing.py
│   │   ├── partition.py
│   │   └── dataset.py
│   ├── models/
│   │   └── __init__.py
│   ├── fl/
│   │   └── __init__.py
│   ├── privacy/
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
├── notebooks/
├── tests/
│   └── test_data_pipeline.py
├── experiments/
├── outputs/
│   ├── figures/
│   └── checkpoints/
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

Create `requirements.txt` with exact pinned versions:
```
torch>=2.1.0
torchvision>=0.16.0
flwr>=1.7.0
opacus>=1.4.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
pyyaml>=6.0
kagglehub>=0.3.0
tqdm>=4.65.0
grad-cam>=1.4.8
wandb>=0.15.0
jupyter>=1.0.0
pytest>=7.4.0
```

Create `configs/data_config.yaml`:
```yaml
dataset:
  name: "brain-tumor-mri"
  kaggle_slug: "masoudnickparvar/brain-tumor-mri-dataset"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  image_size: 224
  num_classes: 4
  classes: ["glioma", "meningioma", "notumor", "pituitary"]
  
preprocessing:
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  augmentation:
    random_horizontal_flip: 0.5
    random_rotation: 15
    color_jitter:
      brightness: 0.2
      contrast: 0.2

federation:
  num_clients: 3
  dirichlet_alpha: [0.1, 0.5, 1.0]
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

Create `.gitignore` with entries for:
```
data/
outputs/checkpoints/
__pycache__/
*.pyc
.env
wandb/
*.egg-info/
dist/
build/
.ipynb_checkpoints/
```
</action>

<acceptance_criteria>
- `requirements.txt` contains `torch`, `flwr`, `opacus`, `kagglehub`, `grad-cam`
- `configs/data_config.yaml` contains `kaggle_slug: "masoudnickparvar/brain-tumor-mri-dataset"`
- `configs/data_config.yaml` contains `num_classes: 4`
- `configs/data_config.yaml` contains `dirichlet_alpha: [0.1, 0.5, 1.0]`
- Directory `src/data/` exists with `__init__.py`
- Directory `src/models/` exists with `__init__.py`
- Directory `src/fl/` exists with `__init__.py`
- Directory `src/privacy/` exists with `__init__.py`
- `.gitignore` contains `data/`
</acceptance_criteria>
</task>

<task id="1.2" title="Dataset Download Script">
<read_first>
- configs/data_config.yaml
</read_first>

<action>
Create `src/data/download.py` that:

1. Uses `kagglehub` to download the Brain Tumor MRI Dataset:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
   ```
2. Organizes the downloaded data into `data/raw/Training/` and `data/raw/Testing/`
3. Validates the download:
   - Expected ~5,712 training images and ~1,311 testing images
   - 4 subdirectories per split: `glioma/`, `meningioma/`, `notumor/`, `pituitary/`
4. Prints dataset statistics:
   - Total images per class
   - Image dimension statistics (min/max/mean resolution)
   - File format verification (all .jpg)
5. Creates a `data/raw/metadata.json` with download timestamp and counts

Include a `__main__` block so it can be run as:
```bash
python -m src.data.download
```
</action>

<acceptance_criteria>
- `src/data/download.py` contains `kagglehub.dataset_download`
- `src/data/download.py` contains `masoudnickparvar/brain-tumor-mri-dataset`
- `src/data/download.py` contains a function `download_dataset()` or `main()`
- Running `python -m src.data.download` downloads data to `data/raw/`
- Script prints count of images per class
</acceptance_criteria>
</task>

<task id="1.3" title="Data Preprocessing Pipeline">
<read_first>
- configs/data_config.yaml
- src/data/download.py
</read_first>

<action>
Create `src/data/preprocessing.py` with:

1. **`get_train_transforms(image_size=224)`** → returns `torchvision.transforms.Compose`:
   - `Resize((image_size, image_size))`
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomRotation(degrees=15)`
   - `ColorJitter(brightness=0.2, contrast=0.2)`
   - `ToTensor()`
   - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

2. **`get_eval_transforms(image_size=224)`** → returns `torchvision.transforms.Compose`:
   - `Resize((image_size, image_size))`
   - `ToTensor()`
   - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

3. **`preprocess_dataset(raw_dir, processed_dir)`** → copies and optionally resizes images, creates a CSV manifest file `data/processed/manifest.csv` with columns: `filepath`, `label`, `original_split` (train/test), `width`, `height`
</action>

<acceptance_criteria>
- `src/data/preprocessing.py` contains `def get_train_transforms`
- `src/data/preprocessing.py` contains `def get_eval_transforms`
- `src/data/preprocessing.py` contains `Normalize(mean=[0.485, 0.456, 0.406]`
- `src/data/preprocessing.py` contains `RandomHorizontalFlip`
- `src/data/preprocessing.py` contains `RandomRotation`
</acceptance_criteria>
</task>

<task id="1.4" title="Non-IID Data Partitioning for Federated Learning">
<read_first>
- configs/data_config.yaml
- src/data/preprocessing.py
</read_first>

<action>
Create `src/data/partition.py` with:

1. **`dirichlet_partition(labels, num_clients, alpha, seed=42)`**:
   - Uses Dirichlet distribution to create non-IID splits
   - `alpha=0.1` → highly non-IID (each client gets mostly 1-2 classes)
   - `alpha=1.0` → approximately IID
   - Returns `List[List[int]]` — list of sample indices per client
   - Ensures every client gets at least `min_samples=10` per assigned class

2. **`pathological_partition(labels, num_clients, classes_per_client=2, seed=42)`**:
   - Each client gets only `classes_per_client` classes
   - Simulates extreme specialization (e.g., Hospital A only sees gliomas)

3. **`iid_partition(labels, num_clients, seed=42)`**:
   - Simple uniform random split (baseline)

4. **`visualize_partition(partition_indices, labels, class_names, save_path)`**:
   - Creates a stacked bar chart showing class distribution per client
   - Saves to `outputs/figures/partition_{alpha}.png`

5. **`create_federated_datasets(data_dir, num_clients, alpha, val_ratio=0.15, test_ratio=0.15)`**:
   - Combines partitioning + dataset creation
   - Returns dict: `{client_id: {"train": Dataset, "val": Dataset, "test": Dataset}}`
   - Also returns a global test set for centralized evaluation
</action>

<acceptance_criteria>
- `src/data/partition.py` contains `def dirichlet_partition`
- `src/data/partition.py` contains `numpy.random.dirichlet`
- `src/data/partition.py` contains `def pathological_partition`
- `src/data/partition.py` contains `def iid_partition`
- `src/data/partition.py` contains `def visualize_partition`
- `src/data/partition.py` contains `alpha` parameter in `dirichlet_partition`
</acceptance_criteria>
</task>

<task id="1.5" title="PyTorch Dataset Class">
<read_first>
- src/data/preprocessing.py
- src/data/partition.py
</read_first>

<action>
Create `src/data/dataset.py` with:

1. **`BrainTumorDataset(torch.utils.data.Dataset)`**:
   - `__init__(self, image_paths, labels, transform=None)`
   - `__getitem__` loads image with PIL, applies transform, returns (image, label)
   - `__len__` returns number of samples
   - Class mapping: `{"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}`

2. **`get_dataloaders(datasets_dict, batch_size=32, num_workers=4)`**:
   - Takes output of `create_federated_datasets()`
   - Returns dict of DataLoaders per client per split
   - Train loaders: `shuffle=True`, `drop_last=True`
   - Val/Test loaders: `shuffle=False`

3. **`get_centralized_dataloaders(data_dir, batch_size=32)`**:
   - Loads all data as single train/val/test split (for centralized baseline)
   - Uses standard 70/15/15 split with stratification
</action>

<acceptance_criteria>
- `src/data/dataset.py` contains `class BrainTumorDataset`
- `src/data/dataset.py` contains `torch.utils.data.Dataset` in class definition
- `src/data/dataset.py` contains `def __getitem__`
- `src/data/dataset.py` contains `def get_dataloaders`
- `src/data/dataset.py` contains `def get_centralized_dataloaders`
- `src/data/dataset.py` contains class mapping with `glioma` and `pituitary`
</acceptance_criteria>
</task>

<task id="1.6" title="Unit Tests for Data Pipeline">
<read_first>
- src/data/partition.py
- src/data/dataset.py
- src/data/preprocessing.py
</read_first>

<action>
Create `tests/test_data_pipeline.py` with pytest tests:

1. `test_dirichlet_partition_correct_total()` — total indices across all clients equals total samples
2. `test_dirichlet_partition_no_overlap()` — no sample appears in multiple clients
3. `test_dirichlet_noniid_skew()` — with α=0.1, at least one client has >60% of one class
4. `test_iid_partition_balance()` — each client gets ~equal samples (within 10%)
5. `test_transforms_output_shape()` — train transforms produce tensor of shape (3, 224, 224)
6. `test_dataset_len()` — BrainTumorDataset reports correct length
7. `test_dataset_label_range()` — all labels in {0, 1, 2, 3}

Use `@pytest.fixture` for shared test data (create dummy images with PIL).
</action>

<acceptance_criteria>
- `tests/test_data_pipeline.py` contains `def test_dirichlet_partition_correct_total`
- `tests/test_data_pipeline.py` contains `def test_dirichlet_partition_no_overlap`
- `tests/test_data_pipeline.py` contains `import pytest`
- `tests/test_data_pipeline.py` contains at least 5 test functions
- `pytest tests/test_data_pipeline.py` exits with code 0
</acceptance_criteria>
</task>

<task id="1.7" title="EDA Notebook">
<read_first>
- src/data/download.py
- src/data/partition.py
</read_first>

<action>
Create `notebooks/01_eda.ipynb` (as a Python script that can be converted) or `notebooks/01_eda.py` with:

1. **Dataset Overview**: Total images, per-class counts, class imbalance ratio
2. **Sample Visualization**: 4×4 grid of random samples per class
3. **Image Statistics**: Histogram of image dimensions, aspect ratios, pixel intensity distributions
4. **Class Distribution**: Pie chart and bar chart of class frequencies
5. **Non-IID Visualization**: Side-by-side partition plots for α=0.1, 0.5, 1.0
6. **Per-Client Statistics Table**: Number of samples per client per class

Save all figures to `outputs/figures/eda/`
</action>

<acceptance_criteria>
- `notebooks/01_eda.py` or `notebooks/01_eda.ipynb` exists
- Script references all 4 classes: glioma, meningioma, notumor, pituitary
- Script creates at least 3 visualization outputs
- Script imports from `src.data.partition`
</acceptance_criteria>
</task>

## Verification

<must_haves>
1. `requirements.txt` has all ML/FL dependencies (torch, flwr, opacus)
2. Dataset can be downloaded via `python -m src.data.download`
3. Non-IID partitioning works for 3 clients with Dirichlet α=0.1
4. All 7 unit tests pass: `pytest tests/test_data_pipeline.py`
5. EDA notebook produces partition visualizations
</must_haves>
