---
phase: 5
plan: 01
title: "Fairness Analysis & Explainability"
wave: 1
depends_on: ["phase-3", "phase-4"]
files_modified:
  - src/utils/fairness.py
  - src/utils/gradcam.py
  - scripts/run_fairness_analysis.py
  - scripts/generate_gradcam.py
requirements_addressed: [REQ-005]
autonomous: true
---

# Phase 5: Fairness Analysis & Explainability

<objective>
Evaluate per-site fairness of the federated model using Jain fairness index, analyze model bias across tumor types per site, and generate Grad-CAM attention visualizations for clinical interpretability.
</objective>

## Tasks

<task id="5.1" title="Fairness Metrics Implementation">
<read_first>
- src/utils/metrics.py
- src/fl/utils.py
</read_first>

<action>
Create `src/utils/fairness.py`:

1. **`jain_fairness_index(accuracies: List[float])`** → float in [0, 1]:
   ```python
   J = (sum(x)**2) / (n * sum(x**2))
   ```
   - J=1.0 means perfect fairness (all sites equal performance)

2. **`compute_site_metrics(model, site_dataloaders, device)`**:
   - Evaluates global model on each site's test set
   - Returns dict: `{site_id: {"accuracy": ..., "auc": ..., "f1": ..., "per_class_acc": {...}}}`

3. **`fairness_gap(site_metrics)`** → max_accuracy - min_accuracy across sites

4. **`bias_analysis(site_metrics, class_names)`**:
   - Identifies which tumor types are under-served at which site
   - Returns a DataFrame: sites × classes with accuracy

5. **`plot_fairness_heatmap(site_metrics, class_names, save_path)`**:
   - Sites vs classes heatmap (seaborn with annotation)
   - Shows where the model performs well/poorly per site per tumor type

6. **`plot_fairness_comparison(fedavg_metrics, fedprox_metrics, scaffold_metrics, save_path)`**:
   - Grouped bar chart comparing Jain index across strategies
   - **Key Paper Figure**
</action>

<acceptance_criteria>
- `src/utils/fairness.py` contains `def jain_fairness_index`
- `src/utils/fairness.py` contains Jain formula `sum(x)**2 / (n * sum(x**2))`
- `src/utils/fairness.py` contains `def fairness_gap`
- `src/utils/fairness.py` contains `def plot_fairness_heatmap`
- `src/utils/fairness.py` contains `seaborn` import
</acceptance_criteria>
</task>

<task id="5.2" title="Grad-CAM Explainability">
<read_first>
- src/models/resnet.py
- src/data/dataset.py
</read_first>

<action>
Create `src/utils/gradcam.py`:

1. **`generate_gradcam(model, image_tensor, target_class, target_layer)`**:
   - Uses `pytorch_grad_cam.GradCAM` from the `grad-cam` library
   - For ResNet-50: `target_layer = model.layer4[-1]`
   - Returns heatmap overlay on original image

2. **`generate_gradcam_grid(model, dataloader, class_names, num_samples=4, save_path)`**:
   - Creates a 4×4 grid: rows = tumor classes, columns = different samples
   - Left: original image, Right: Grad-CAM overlay
   - **Key Paper Figure** for interpretability section

3. **`compare_gradcam_federated_vs_centralized(fl_model, centralized_model, images, save_path)`**:
   - Side-by-side Grad-CAM comparison
   - Shows whether FL model attends to same regions as centralized model
</action>

<acceptance_criteria>
- `src/utils/gradcam.py` contains `GradCAM`
- `src/utils/gradcam.py` contains `def generate_gradcam`
- `src/utils/gradcam.py` contains `def generate_gradcam_grid`
- `src/utils/gradcam.py` contains `layer4` reference for ResNet
</acceptance_criteria>
</task>

<task id="5.3" title="Fairness & Explainability Runner Script">
<read_first>
- src/utils/fairness.py
- src/utils/gradcam.py
</read_first>

<action>
Create `scripts/run_fairness_analysis.py`:
1. Loads trained FL models (FedAvg, FedProx) + centralized baseline
2. Evaluates each on per-site test data
3. Computes: Jain index, fairness gap, per-site per-class accuracy
4. Generates all figures: fairness heatmap, comparison bar chart
5. Saves results table to `outputs/fairness/results.csv`

Create `scripts/generate_gradcam.py`:
1. Loads best FL model
2. Generates Grad-CAM grid for each tumor class
3. Generates federated vs. centralized comparison
4. Saves to `outputs/figures/gradcam/`
</action>

<acceptance_criteria>
- `scripts/run_fairness_analysis.py` loads FL and centralized models
- `scripts/run_fairness_analysis.py` computes `jain_fairness_index`
- `scripts/generate_gradcam.py` saves figures to `outputs/figures/gradcam/`
- Scripts generate at least 3 publication-quality figures
</acceptance_criteria>
</task>

## Verification

<must_haves>
1. Jain fairness index computed for at least 2 FL strategies
2. Fairness gap < 5% for FedProx with α=0.5
3. Grad-CAM grid generated for all 4 tumor classes
4. Per-site accuracy heatmap saved
5. Federated vs Centralized Grad-CAM comparison generated
</must_haves>
