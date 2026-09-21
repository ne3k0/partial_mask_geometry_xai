"""
compute_projections.py — logit-space trajectory decomposition.

Reads the sigmoid outputs produced by data_extract.py and computes the
FOC→original projection for each partial mask, saving one NPZ per (clip, fill):

    {VP_ROOT}/{model}/{cls}/{clip_id}_{fill}.npz
        d       : (527,)  float32 — attribution axis z_orig − z_foc
        t       : (9600,) float32 — scalar projection, 0 at FOC, 1 at original
        d_perp  : (9600,) float32 — off-axis residual magnitude
        d_total : (9600,) float32 — total displacement from FOC in logit space

These files are consumed by:
    notebooks/logit_geometry.ipynb          → Figure 2
    notebooks/off_axis_analysis/*.ipynb     → Table 3

Input layout (produced by data_extract.py):
    {DATA_ROOT}/{model}/{cls}/embs/{clip_id}_original_sigmoid.npy   (527,)
    {DATA_ROOT}/{model}/{cls}/embs/{clip_id}_foc_{fill}.npy         (527,)
    {DATA_ROOT}/{model}/{cls}/embs/{clip_id}_perturb_{fill}.npy     (N_masks, 527)

Usage (run from within partial_mask_geometry_xai/ with freqrex-env):
    python compute_projections.py \\
        --data-root /path/to/data/eval_data \\
        --vp-root   /path/to/data/vector_projs

    # Restrict to one model or class:
    python compute_projections.py ... --model panns_no_specaug
    python compute_projections.py ... --class Bagpipes
"""

import argparse
import os
import numpy as np

FILLS   = ["zero", "mean", "gaussian_noise"]
MODELS  = ["panns_no_specaug", "panns_specaug_trained", "ast_wrapper"]
CLASSES = [
    "bagpipes", "boing", "chicken_rooster", "didgeridoo", "dog", "drum_kit",
    "frog", "frying_food", "gunshot_gunfire", "hair_dryer", "harmonica",
    "heart_sounds_heartbeat", "insect", "owl", "rain", "rub", "sewing_machine",
    "speech", "thunder", "timpani", "train", "whispering",
]

EPS_LOGIT = 1e-7   # clamp before logit: keeps values in (-16.1, 16.1)
EPS_NORM  = 1e-10  # minimum ||d||² — below this z_orig ≈ z_foc, axis degenerate


def sigmoid_to_logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(q.astype(np.float64), EPS_LOGIT, 1.0 - EPS_LOGIT)
    return np.log(q / (1.0 - q))


def project(z_orig: np.ndarray, z_foc: np.ndarray, Z: np.ndarray):
    """
    Decompose each mask's logit vector relative to the FOC→original axis.

    Parameters
    ----------
    z_orig : (527,)      float64 — logits for unoccluded clip
    z_foc  : (527,)      float64 — logits under full occlusion
    Z      : (N, 527)    any     — logits for all N partial masks

    Returns (d, t, d_perp, d_total) or None if the attribution axis is degenerate
    (z_orig ≈ z_foc, i.e. full occlusion has no effect on this clip/fill).

    d       : (527,) float32 — reference axis (unnormalised)
    t       : (N,)   float32 — scalar projection: 0 at FOC, 1 at original
    d_perp  : (N,)   float32 — off-axis residual magnitude
    d_total : (N,)   float32 — total displacement from FOC
    """
    d    = z_orig - z_foc            # (527,) attribution axis
    d_sq = float(np.dot(d, d))       # ||d||²
    if d_sq < EPS_NORM:
        return None

    W       = Z.astype(np.float64) - z_foc      # (N, 527) displacement from FOC
    t       = (W @ d) / d_sq                    # (N,) scalar projections
    d_total = np.linalg.norm(W, axis=1)         # (N,) total displacement

    # Pythagorean identity: avoids materialising (N, 527) residual matrix in memory
    d_perp = np.sqrt(np.maximum(0.0, d_total ** 2 - t ** 2 * d_sq))

    return (
        d.astype(np.float32),
        t.astype(np.float32),
        d_perp.astype(np.float32),
        d_total.astype(np.float32),
    )


def is_valid_npz(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with np.load(path) as f:
            assert set(f.files) >= {"d", "t", "d_perp", "d_total"}
            assert f["t"].ndim == 1
        return True
    except Exception:
        return False


def compute_for_clip(embs_dir: str, clip_id: str, dest_dir: str) -> dict:
    """
    Compute and save projection NPZs for all fills for one clip.

    Returns a dict with keys 'saved', 'skipped', 'degenerate', 'errors' (all counts).
    """
    counts = {"saved": 0, "skipped": 0, "degenerate": 0, "errors": 0}

    orig_path = os.path.join(embs_dir, f"{clip_id}_original_sigmoid.npy")
    if not os.path.exists(orig_path):
        print(f"  MISSING original: {orig_path}")
        counts["errors"] += len(FILLS)
        return counts

    z_orig = sigmoid_to_logit(np.load(orig_path, allow_pickle=False))

    written_this_clip = []
    for fill in FILLS:
        out_path = os.path.join(dest_dir, f"{clip_id}_{fill}.npz")

        if is_valid_npz(out_path):
            counts["skipped"] += 1
            continue

        # Remove any corrupt partial file before attempting to write
        if os.path.exists(out_path):
            os.remove(out_path)

        foc_path     = os.path.join(embs_dir, f"{clip_id}_foc_{fill}.npy")
        perturb_path = os.path.join(embs_dir, f"{clip_id}_perturb_{fill}.npy")

        for p in [foc_path, perturb_path]:
            if not os.path.exists(p):
                print(f"  MISSING: {p}")
                counts["errors"] += 1
                break
        else:
            try:
                z_foc = sigmoid_to_logit(np.load(foc_path, allow_pickle=False))
                Z     = sigmoid_to_logit(np.load(perturb_path, allow_pickle=False,
                                                  mmap_mode="r"))

                result = project(z_orig, z_foc, Z)
                if result is None:
                    print(f"  DEGENERATE (||d||≈0): {clip_id}/{fill} — skipping")
                    counts["degenerate"] += 1
                    continue

                d, t, d_perp, d_total = result
                np.savez(out_path, d=d, t=t, d_perp=d_perp, d_total=d_total)
                written_this_clip.append(out_path)
                counts["saved"] += 1

            except Exception as e:
                # Roll back any files written for this clip on error
                for fp in written_this_clip:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
                print(f"  ERROR {clip_id}/{fill}: {e}")
                counts["errors"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Compute logit-space trajectory decompositions from data_extract.py output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-root", required=True,
                        help="Root of data_extract.py output: {model}/{cls}/embs/")
    parser.add_argument("--vp-root",   required=True,
                        help="Output root for VP NPZs: {model}/{cls}/{clip}_{fill}.npz")
    parser.add_argument("--model",     default=None,
                        help="Restrict to one model (e.g. panns_no_specaug)")
    parser.add_argument("--class",     dest="cls", default=None,
                        help="Restrict to one class slug (e.g. bagpipes)")
    args = parser.parse_args()

    models  = [args.model] if args.model else MODELS
    classes = [args.cls]   if args.cls   else CLASSES

    total = {"saved": 0, "skipped": 0, "degenerate": 0, "errors": 0}

    for model in models:
        for cls in classes:
            embs_dir = os.path.join(args.data_root, model, cls, "embs")
            dest_dir = os.path.join(args.vp_root,   model, cls)

            if not os.path.isdir(embs_dir):
                print(f"SKIP (no embs dir): {model}/{cls}")
                continue

            os.makedirs(dest_dir, exist_ok=True)

            clip_ids = sorted(
                f[: -len("_original_sigmoid.npy")]
                for f in os.listdir(embs_dir)
                if f.endswith("_original_sigmoid.npy")
            )
            if not clip_ids:
                print(f"SKIP (no clips): {model}/{cls}")
                continue

            print(f"\n{model}/{cls}  ({len(clip_ids)} clips)")
            for clip_id in clip_ids:
                counts = compute_for_clip(embs_dir, clip_id, dest_dir)
                for k in total:
                    total[k] += counts[k]
                if counts["saved"]:
                    print(f"  {clip_id}  saved={counts['saved']} "
                          f"skipped={counts['skipped']} "
                          f"degen={counts['degenerate']} "
                          f"err={counts['errors']}")

    print(f"\nDone.  saved={total['saved']}  skipped={total['skipped']}  "
          f"degenerate={total['degenerate']}  errors={total['errors']}")
    print(f"VP root: {args.vp_root}")


if __name__ == "__main__":
    main()
