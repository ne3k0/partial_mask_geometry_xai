# partial_mask_geometry_xai

Logit-space trajectory decomposition of audio classifier outputs under partial masking. Companion code for *Mask-Induced Displacement in Audio XAI via Logit Trajectory Decomposition*.

## Pipeline

```
data_extract.py  ->  projection.py  ->  notebooks/
```

| Script | Role |
|--------|------|
| `data_extract.py` | GPU. Generates partially masked spectrograms and runs model inference on each, saving sigmoid outputs and masks (optional: intermediate layer representations and attribution maps, not used in this paper). |
| `projection.py` | CPU. Converts sigmoid probabilities to logits; decomposes each mask's displacement into on-axis (original -> fully occluded) and off-axis components. |
| `notebooks/` | Analysis and figures. Reads projection outputs; no GPU required. |
| `src/partial_masks/` | Core library: model wrappers, perturbation & masking. |

## Setup

```bash
conda create -n partial_mask_geometry python=3.11
conda activate partial_mask_geometry
pip install -r requirements.txt
```

## Data layout

```
{data-root}/
  {model}/{class}/embs/
    {clip}_original_sigmoid.npy          # (527,)   sigmoid — original clip
    {clip}_foc_sigmoid_{fill}.npy        # (527,)   sigmoid — fully occluded
    {clip}_perturb_sigmoid_{fill}.npy    # (N,527)  sigmoid — per-mask outputs
    {clip}_perturbations_{fill}.npz      # occ_fracs (N,) + layer activations
```

Evaluated models: `ast_wrapper`, `panns_no_specaug`, `panns_specaug_trained`. Additional models can be integrated by adding a wrapper to `src/partial_masks/model_wrappers/`.  
Implemented fills: `zero`, `mean`, `gaussian_noise`. Additional fills can be added in `src/partial_masks/perturbation.py`.

## Usage

**Extract embeddings** (GPU node):
```bash
python data_extract.py --manifest data/eval_manifest.json \
    --model ast_wrapper --output-root /path/to/eval_data
```

**Compute projections** (CPU):
```bash
python projection.py \
    --data-root /path/to/eval_data \
    --vp-root   /path/to/vector_projs
```

Filter by model, class, or fill:
```bash
python projection.py --model ast_wrapper --class bagpipes --fills zero mean
```

## Notebooks

| Notebook | Output |
|----------|--------|
| `fully_occluded_condition.ipynb` | FOC fingerprint table, per-model top-1/top-5 confidence, consistency analysis. |
| `logit_geometry.ipynb` | On-axis scalar projections \tau and off-axis norms d_perp across retention bins. |
| `off_axis_direction.ipynb` | Covariance of off-axis vectors v_perp; eigenanalysis and participation ratio per (model, fill). |

Set `DATA_ROOT`, `VP_ROOT`, and `OUTPUT_DIR` in cell 1 of each notebook before running.
