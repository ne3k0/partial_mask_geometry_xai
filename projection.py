"""
Logit-space trajectory decomposition for partial mask outputs.

Usage:
    python projection.py --data-root /path/to/eval_data --vp-root /path/to/vector_projs
    python projection.py ... --model panns_no_specaug --class bagpipes --fills zero mean

Supports two extraction formats (auto-detected per class directory):
  legacy  {clip}_original_logits.npy, {clip}_fully_occluded_{fill}_logits.npy,
          {clip}_perturbations_{fill}.npz  (keys: logits, occ_fracs)
          NOTE: all three 'logits' sources are sigmoid confidences despite the name.
  new     {clip}_original_sigmoid.npy, {clip}_foc_{fill}.npy,
          {clip}_perturb_{fill}.npy, {clip}_perturbations_{fill}.npz
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

EPS_LOGIT = 1e-7
EPS_NORM  = 1e-10


def to_logit(q):
    q = np.clip(q.astype(np.float64), EPS_LOGIT, 1.0 - EPS_LOGIT)
    return np.log(q / (1.0 - q))


def project(z_orig, z_foc, Z):
    """Returns (d, t, d_perp, d_total, v_perp) or None if z_orig ≈ z_foc."""
    d    = z_orig - z_foc
    d_sq = float(np.dot(d, d))
    if d_sq < EPS_NORM:
        return None
    W       = Z - z_foc
    t       = (W @ d) / d_sq
    d_total = np.linalg.norm(W, axis=1)
    # Pythagorean identity — avoids a second (N, 527) allocation for the residual norm
    d_perp  = np.sqrt(np.maximum(0.0, d_total**2 - t**2 * d_sq))
    v_perp  = (W - t[:, np.newaxis] * d[np.newaxis, :]).astype(np.float32)
    return d.astype(np.float32), t.astype(np.float32), d_perp.astype(np.float32), d_total.astype(np.float32), v_perp


def _valid(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with np.load(path) as f:
            return {"d", "t", "d_perp", "d_total", "v_perp", "ret"} <= set(f.files) and f["t"].ndim == 1
    except Exception:
        return False


def _detect_format(embs_dir):
    """Return 'legacy' if old extraction format is present, else 'new'."""
    files = os.listdir(embs_dir)
    if any(f.endswith("_original_logits.npy") for f in files):
        return "legacy"
    return "new"


def _load_orig(embs_dir, clip_id, fmt):
    """Load and logit-transform the original clip's sigmoid output. Returns None if missing."""
    suffix = "_original_logits.npy" if fmt == "legacy" else "_original_sigmoid.npy"
    p = os.path.join(embs_dir, f"{clip_id}{suffix}")
    if not os.path.exists(p):
        return None
    return to_logit(np.load(p, allow_pickle=False))


def _load_fill_legacy(embs_dir, clip_id, fill):
    """Load FOC + perturbation data from legacy Apocrita format.

    Files use misleading '_logits' / 'logits' naming but contain sigmoid
    confidences — to_logit() is applied to all arrays.
    """
    foc_p  = os.path.join(embs_dir, f"{clip_id}_fully_occluded_{fill}_logits.npy")
    pert_p = os.path.join(embs_dir, f"{clip_id}_perturbations_{fill}.npz")
    if not os.path.exists(foc_p) or not os.path.exists(pert_p):
        return None
    z_foc = to_logit(np.load(foc_p, allow_pickle=False))
    with np.load(pert_p) as npz:
        Z   = to_logit(npz["logits"])          # sigmoid despite name
        ret = (1.0 - npz["occ_fracs"]).astype(np.float32)
    return z_foc, Z, ret


def _load_fill_new(embs_dir, clip_id, fill):
    """Load FOC + perturbation data from new data_extract.py format."""
    foc_p  = os.path.join(embs_dir, f"{clip_id}_foc_{fill}.npy")
    per_p  = os.path.join(embs_dir, f"{clip_id}_perturb_{fill}.npy")
    pert_p = os.path.join(embs_dir, f"{clip_id}_perturbations_{fill}.npz")
    if not os.path.exists(foc_p) or not os.path.exists(per_p) or not os.path.exists(pert_p):
        return None
    z_foc = to_logit(np.load(foc_p,  allow_pickle=False))
    Z     = to_logit(np.load(per_p,  allow_pickle=False))
    with np.load(pert_p) as npz:
        ret = (1.0 - npz["occ_fracs"]).astype(np.float32)
    return z_foc, Z, ret


def process_clip(embs_dir, clip_id, dest_dir, fills, fmt):
    counts      = dict(saved=0, skipped=0, degenerate=0, errors=0)
    fill_loader = _load_fill_legacy if fmt == "legacy" else _load_fill_new

    z_orig = _load_orig(embs_dir, clip_id, fmt)
    if z_orig is None:
        print(f"  MISSING orig: {clip_id}")
        counts["errors"] += len(fills)
        return counts

    for fill in fills:
        out = os.path.join(dest_dir, f"{clip_id}_{fill}.npz")
        if _valid(out):
            counts["skipped"] += 1
            continue
        if os.path.exists(out):
            os.remove(out)

        try:
            fill_data = fill_loader(embs_dir, clip_id, fill)
            if fill_data is None:
                print(f"  MISSING {clip_id}/{fill}")
                counts["errors"] += 1
                continue
            z_foc, Z, ret = fill_data
            if len(ret) != len(Z):
                raise ValueError(f"occ_fracs length {len(ret)} != Z length {len(Z)} — partial write?")
            result = project(z_orig, z_foc, Z)
            if result is None:
                print(f"  DEGENERATE {clip_id}/{fill}")
                counts["degenerate"] += 1
                continue
            d, t, d_perp, d_total, v_perp = result
            np.savez(out, d=d, t=t, d_perp=d_perp, d_total=d_total, v_perp=v_perp, ret=ret)
            counts["saved"] += 1
        except Exception as e:
            if os.path.exists(out):
                try: os.remove(out)
                except OSError: pass
            print(f"  ERROR {clip_id}/{fill}: {e}")
            counts["errors"] += 1

    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--vp-root",   required=True)
    ap.add_argument("--model",     default=None, choices=MODELS)
    ap.add_argument("--class",     dest="cls", default=None, choices=CLASSES)
    ap.add_argument("--fills",     nargs="+", default=None, choices=FILLS)
    args = ap.parse_args()

    models  = [args.model] if args.model else MODELS
    classes = [args.cls]   if args.cls   else CLASSES
    fills   = args.fills   if args.fills else FILLS
    total   = dict(saved=0, skipped=0, degenerate=0, errors=0)

    for model in models:
        for cls in classes:
            embs_dir = os.path.join(args.data_root, model, cls, "embs")
            dest_dir = os.path.join(args.vp_root,   model, cls)
            if not os.path.isdir(embs_dir):
                continue
            os.makedirs(dest_dir, exist_ok=True)

            fmt    = _detect_format(embs_dir)
            suffix = "_original_logits.npy" if fmt == "legacy" else "_original_sigmoid.npy"
            clips  = sorted(f[:-len(suffix)] for f in os.listdir(embs_dir) if f.endswith(suffix))
            if not clips:
                continue
            print(f"\n{model}/{cls}  ({len(clips)} clips)  [{fmt}]")
            for clip_id in clips:
                c = process_clip(embs_dir, clip_id, dest_dir, fills, fmt)
                for k in total: total[k] += c[k]
                if c["saved"]:
                    print(f"  {clip_id}  saved={c['saved']} skip={c['skipped']} degen={c['degenerate']} err={c['errors']}")

    print(f"\nDone. saved={total['saved']} skipped={total['skipped']} degenerate={total['degenerate']} errors={total['errors']}")


if __name__ == "__main__":
    main()
