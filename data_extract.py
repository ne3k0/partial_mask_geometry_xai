"""
data_extract.py — STFT perturbation sweep for partial mask geometry analysis.

Always saves: original + FOC sigmoid probs, per-mask sigmoid probs + occ_fracs, samples.csv, summary.csv.
Optional (all on by default): --no-layers, --no-attribution, --no-masks.
Masks are deterministic and shared across fills.

Usage:
    python data_extract.py <model_wrapper> <class_name> --wav-dir DIR --out-dir DIR [options]
"""

import argparse
import csv
import importlib
import json
import os
import sys
import time
import traceback
import warnings

import librosa
import numpy as np
from numba import njit, prange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from partial_masks.model_stft import get_stft_helper
from partial_masks.config import N_SEEDS, L, K_F, K_T, P
from partial_masks.perturbation import (
    FILL_FACTORIES,
    generate_block_boundaries,
    build_cell_masks,
    pack_ragged_arrays,
    expand_binary,
)


# Attribution

@njit(parallel=True)
def calc_marginal_importance(all_masks, all_confidences, n_cells):
    """phi[j] = mean(f | cell j on) - mean(f | cell j off), pooled across all seeds."""
    N_s = len(all_confidences)
    phi = np.zeros(n_cells, dtype=np.float64)
    for j in prange(n_cells):
        sum_on, count_on   = 0.0, 0
        sum_off, count_off = 0.0, 0
        for i in range(N_s):
            if all_masks[i, j]:
                sum_on  += all_confidences[i]; count_on  += 1
            else:
                sum_off += all_confidences[i]; count_off += 1
        mean_on  = sum_on  / count_on  if count_on  > 0 else 0.0
        mean_off = sum_off / count_off if count_off > 0 else 0.0
        phi[j] = mean_on - mean_off
    return phi


# Model loading

def load_model(wrapper_name):
    name = wrapper_name.replace(".py", "")
    mod = importlib.import_module(f"partial_masks.model_wrappers.{name}")
    return mod.model


def load_audio(path, sr):
    signal, _ = librosa.load(path, sr=sr, mono=True)
    return signal.astype(np.float32)


# Anchor forward pass (original α = 1 and FOC α = 0)

def query_anchor(model, stft_helper, Zxx, target, fill_fn, stft_mask, save_layers=False):
    """
    One forward pass for an anchor condition (α=0 FOC or α=1 original).
    stft_mask : (F, T) — ones = retain, zeros = occlude.
    Returns (conf, layer_acts, preds[0]). layer_acts always has sigmoid (527,) float32;
    also has layer_0 … penultimate when save_layers=True.
    """
    Zxx_filled = fill_fn(Zxx, stft_mask) # apply fill if occluded
    log_mel = stft_helper.to_logmel(Zxx_filled) # convert to log mel spectrogram
    if save_layers:
        preds, raw_acts = model.infer_all_layers_from_mel(log_mel[np.newaxis])         # full layer collection
        layer_acts = {k: v[0].astype(np.float32, copy=False)
                      for k, v in raw_acts.items()}                                    # squeeze batch dim → (C,) per key; copy=False avoids redundant alloc when already float32
    else:
        preds, prob_vec = model.infer_sigmoid_from_mel(log_mel[np.newaxis])            # sigmoid only, no hidden states
        layer_acts = {"sigmoid": prob_vec[0].astype(np.float32, copy=False)}
    conf = float(layer_acts["sigmoid"][model.class_to_idx[target]])                    # GT-class confidence scalar — direct index, no label scan
    return conf, layer_acts, preds[0]


# Perturbation forward pass (0 < α < 1) — batch equivalent of query_anchor

def query_perturb(model, stft_helper, Zxx, target, fill_fn, stft_masks, save_layers=True):
    """
    Batched forward pass for one seed's masks (batch equivalent of query_anchor).
    stft_masks : (n_masks, F, T). Returns (confs (n_masks,), layer_acts {key: (n_masks, C)}).
    """
    n_masks = stft_masks.shape[0]                                                           # batch size (equals module constant L at runtime)
    M = stft_helper.n_mels
    _, T = Zxx.shape
    log_mel_batch = np.empty((n_masks, M, T), dtype=np.float32)                             # pre-allocate; every row overwritten below — np.empty avoids wasted zero-fill
    for i in range(n_masks):
        log_mel_batch[i] = stft_helper.to_logmel(fill_fn(Zxx, stft_masks[i]))              # apply fill per mask → log mel
    if save_layers:
        _, raw_acts = model.infer_all_layers_from_mel(log_mel_batch)                        # full layer collection; preds unused (conf read from sigmoid)
        layer_acts = {k: v.astype(np.float32, copy=False)
                      for k, v in raw_acts.items()}                                         # cast to float32; copy=False avoids redundant alloc when already float32
    else:
        _, prob_vec = model.infer_sigmoid_from_mel(log_mel_batch)                           # sigmoid only, no layer hooks; preds unused
        layer_acts = {"sigmoid": prob_vec.astype(np.float32, copy=False)}
    confs = layer_acts["sigmoid"][:, model.class_to_idx[target]]                            # GT-class scalar per mask — direct numpy index, no label scan
    return confs, layer_acts


def run_sweep(model, stft_helper, Zxx, target, fills_todo, fill_fns, save_layers=True):
    """
    N_SEEDS × L masks per fill (N=9,600 total). Masks shared across fills.
    Returns {fill: {all_masks, all_confs, all_layers, all_occ_fracs}}.
    """
    M   = stft_helper.n_mels
    X   = stft_helper.x
    F, T = Zxx.shape
    X2T  = np.array_split(np.arange(T), X)

    M2F_arr, M2F_len = pack_ragged_arrays(stft_helper.M2F)
    X2T_arr, X2T_len = pack_ragged_arrays(X2T)

    accum            = {fill: {"confs": [], "layers": None} for fill in fills_todo}
    shared_masks     = []   # fill-independent; accumulated once per seed to avoid N_fills-fold duplication
    shared_occ_fracs = []   # fill-independent; same reason

    for seed in range(N_SEEDS):
        t0        = time.perf_counter()
        rng       = np.random.default_rng(seed)
        block_map = generate_block_boundaries(rng, M, X)
        draws     = rng.random((L, K_F * K_T)) < P
        masks     = build_cell_masks(block_map.flatten(), draws)   # (L, M*X) bool
        stft_masks = expand_binary(masks, M, X, M2F_arr, M2F_len,
                                   X2T_arr, X2T_len, F, T, L)     # (L, F, T)
        occ_fracs  = 1.0 - masks.mean(axis=1)                     # (L,)

        shared_masks.append(masks)          # store once per seed, not once per fill
        shared_occ_fracs.append(occ_fracs)  # store once per seed, not once per fill

        for fill_name in fills_todo:
            confs, layer_acts = query_perturb(
                model, stft_helper, Zxx, target,
                fill_fns[fill_name], stft_masks,
                save_layers=save_layers,
            )
            ac = accum[fill_name]
            ac["confs"].append(confs)
            if ac["layers"] is None:
                ac["layers"] = {k: [] for k in layer_acts.keys()}
            elif set(layer_acts) != set(ac["layers"]):
                raise RuntimeError(
                    f"Model returned different layer keys on seed {seed}: "
                    f"expected {set(ac['layers'])}, got {set(layer_acts)}"
                )
            for k in layer_acts:
                ac["layers"][k].append(layer_acts[k])

        print(f"    Seed {seed + 1}/{N_SEEDS}  ({time.perf_counter() - t0:.1f}s)")

    all_masks     = np.vstack(shared_masks)            # (N, n_cells) — allocated once, shared across fill dicts
    all_occ_fracs = np.concatenate(shared_occ_fracs)   # (N,)         — allocated once, shared across fill dicts
    return {
        fill: {
            "all_masks":     all_masks,
            "all_confs":     np.concatenate(accum[fill]["confs"]),
            "all_layers":    {k: np.vstack(v) for k, v in accum[fill]["layers"].items()},
            "all_occ_fracs": all_occ_fracs,
        }
        for fill in fills_todo
    }


# ── Per-clip processing ───────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "clip_id", "model_id", "fill", "target",
    "conf_original", "conf_fully_occluded",
    "focc_top1_label", "focc_top1_score",
    "focc_top2_label", "focc_top2_score",
    "focc_top3_label", "focc_top3_score",
    "n_samples", "n_seeds",
    "original_sigmoid_path", "foc_sigmoid_path", "perturb_npz_path", "attr_map_path",
]

SAMPLES_FIELDS = [
    "clip_id", "model_id", "fill", "target",
    "sample_idx", "seed", "confidence", "occlusion_frac",
    "perturb_npz_path", "row_idx",
]


def _top_k(preds, k=3):
    out = {}
    for i in range(k):
        p = preds[i] if i < len(preds) else {"label": "", "score": 0.0}
        out[f"focc_top{i+1}_label"] = p["label"]
        out[f"focc_top{i+1}_score"] = round(float(p["score"]), 6)
    return out


def process_clip(model, stft_helper, clip_path, out_dirs,
                 summary_writer, samples_writer, class_name, fills_todo, cfg):
    """cfg: {save_layers, save_attribution, save_masks} — all bool."""
    clip_id = os.path.splitext(os.path.basename(clip_path))[0]

    signal = load_audio(clip_path, stft_helper.sr)
    if np.max(np.abs(signal)) == 0 or np.any(np.isnan(signal)):
        raise ValueError("silent or NaN waveform")

    Zxx  = stft_helper.compute_stft(signal)
    F, T = Zxx.shape

    # "zero" always needed for the original α=1 pass regardless of fills_todo
    fills_needed = set(fills_todo) | {"zero"}
    fill_fns = {name: factory(Zxx) for name, factory in FILL_FACTORIES.items()
                if name in fills_needed}

    # Original anchor (α=1)
    conf_orig, orig_acts, _ = query_anchor(
        model, stft_helper, Zxx, class_name,
        fill_fns["zero"], np.ones((F, T), dtype=np.float32),
        save_layers=cfg["save_layers"]
    )
    orig_sigmoid_path = os.path.join(out_dirs["embs"], f"{clip_id}_original_sigmoid.npy")
    np.save(orig_sigmoid_path, orig_acts["sigmoid"])
    if cfg["save_layers"]:
        np.savez_compressed(
            os.path.join(out_dirs["embs"], f"{clip_id}_original_layers.npz"),
            **{k: v for k, v in orig_acts.items() if k != "sigmoid"},
        )
    print(f"    Original conf ({class_name}): {conf_orig:.3f}")

    # FOC anchor (α=0) — one pass per fill
    full_occ_mask = np.zeros((F, T), dtype=np.float32)  # all-zeros mask = full occlusion
    conf_foc, foc_sigmoid_paths, focc_top3 = {}, {}, {}
    for fill_name in fills_todo:
        conf_fo, foc_acts, preds_fo = query_anchor(
            model, stft_helper, Zxx, class_name,
            fill_fns[fill_name], full_occ_mask,
            save_layers=cfg["save_layers"]
        )
        conf_foc[fill_name]  = conf_fo
        focc_top3[fill_name] = preds_fo[:3]
        p = os.path.join(out_dirs["embs"], f"{clip_id}_foc_{fill_name}.npy")
        np.save(p, foc_acts["sigmoid"])
        foc_sigmoid_paths[fill_name] = p
        if cfg["save_layers"]:
            np.savez_compressed(
                os.path.join(out_dirs["embs"], f"{clip_id}_foc_{fill_name}_layers.npz"),
                **{k: v for k, v in foc_acts.items() if k != "sigmoid"},
            )
    print(f"    FOC confs: { {k: f'{v:.3f}' for k, v in conf_foc.items()} }")

    print(f"    Sweep: {N_SEEDS} seeds × {L} masks × {len(fills_todo)} fills ...")
    sweep   = run_sweep(model, stft_helper, Zxx, class_name, fills_todo, fill_fns,
                        save_layers=cfg["save_layers"])
    n_cells = stft_helper.n_mels * stft_helper.x

    # Masks are fill-independent — save once per clip (packed bits)
    if cfg["save_masks"]:
        np.save(os.path.join(out_dirs["masks"], f"{clip_id}_masks.npy"),
                np.packbits(sweep[fills_todo[0]]["all_masks"].astype(np.bool_), axis=1))

    for fill_name in fills_todo:
        sw            = sweep[fill_name]
        all_masks     = sw["all_masks"]      # (9600, n_cells)
        all_confs     = sw["all_confs"]      # (9600,)
        all_layers    = sw["all_layers"]     # always has 'sigmoid' (9600, 527); + layer_N if save_layers
        all_occ_fracs = sw["all_occ_fracs"]  # (9600,)

        # φ_j = E[f | m_j=1] − E[f | m_j=0]
        attr_path = ""
        if cfg["save_attribution"]:
            phi = calc_marginal_importance(all_masks, all_confs.astype(np.float64), n_cells)
            attr_path = os.path.join(out_dirs["attribution_maps"], f"{clip_id}_{fill_name}.npy")
            np.save(attr_path, phi.reshape(stft_helper.n_mels, stft_helper.x))

        # NPZ: occ_fracs + layer arrays (excl. sigmoid, saved separately as flat .npy)
        # confs not saved — recoverable as sigmoid[:, gt_class_idx]
        npz_path = os.path.join(out_dirs["embs"], f"{clip_id}_perturbations_{fill_name}.npz")
        np.savez_compressed(
            npz_path,
            occ_fracs=all_occ_fracs.astype(np.float32),
            **{k: v.astype(np.float32) for k, v in all_layers.items() if k != "sigmoid"},
        )

        # Flat sigmoid .npy — (9600, 527) float32, values in (0,1).
        # Notebooks apply v = ln(q/(1−q)) to enter logit space before decomposition.
        perturb_npy_path = os.path.join(out_dirs["embs"], f"{clip_id}_perturb_{fill_name}.npy")
        np.save(perturb_npy_path, all_layers["sigmoid"].astype(np.float32))

        summary_writer.writerow({
            "clip_id":               clip_id,
            "model_id":              model.model_id,
            "fill":                  fill_name,
            "target":                class_name,
            "conf_original":         round(conf_orig, 6),
            "conf_fully_occluded":   round(conf_foc[fill_name], 6),
            **_top_k(focc_top3[fill_name]),
            "n_samples":             len(all_confs),
            "n_seeds":               N_SEEDS,
            "original_sigmoid_path":  orig_sigmoid_path,
            "foc_sigmoid_path":       foc_sigmoid_paths[fill_name],
            "perturb_npz_path":      npz_path,
            "attr_map_path":         attr_path,
        })

        for idx in range(len(all_confs)):
            samples_writer.writerow({
                "clip_id":        clip_id,
                "model_id":       model.model_id,
                "fill":           fill_name,
                "target":         class_name,
                "sample_idx":     idx,
                "seed":           idx // L,
                "confidence":     round(float(all_confs[idx]), 6),
                "occlusion_frac": round(float(all_occ_fracs[idx]), 6),
                "perturb_npz_path": npz_path,
                "row_idx":        idx,
            })

        print(f"    [{fill_name}] saved. FOC conf={conf_foc[fill_name]:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Perturbation sweep — one model × one class.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_wrapper", help="Wrapper module name, e.g. ast_wrapper")
    parser.add_argument("class_name",   help="AudioSet class label, e.g. Bagpipes")
    parser.add_argument("--wav-dir",    required=True, help="Directory containing .wav clips")
    parser.add_argument("--out-dir",    required=True, help="Root output directory")
    parser.add_argument("--manifest",   default=None,
                        help="Path to eval_manifest.json (default: data/eval_manifest.json)")
    _all_fills = list(FILL_FACTORIES.keys())
    parser.add_argument("--fills",      nargs="+", default=_all_fills,
                        choices=_all_fills,
                        metavar="FILL",
                        help="Fill conditions to run (default: all three)")
    parser.add_argument("--no-layers",      dest="save_layers",      action="store_false",
                        help="Skip layer activations (sigmoid + occ_frac always saved).")
    parser.add_argument("--no-attribution", dest="save_attribution", action="store_false",
                        help="Skip RISE attribution maps.")
    parser.add_argument("--no-masks",       dest="save_masks",       action="store_false",
                        help="Skip packed binary mask files (occ_fracs still saved).")
    parser.set_defaults(save_layers=True, save_attribution=True, save_masks=True)
    args = parser.parse_args()

    cfg = {
        "save_layers":      args.save_layers,
        "save_attribution": args.save_attribution,
        "save_masks":       args.save_masks,
    }

    manifest_path = args.manifest or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "eval_manifest.json"
    )

    print(f"Loading model: {args.model_wrapper} ...")
    model = load_model(args.model_wrapper)
    print(f"Model ID: {model.model_id}")

    stft_helper = get_stft_helper(model.model_id, model)

    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.class_name not in manifest:
        print(f"ERROR: '{args.class_name}' not in manifest. Available: {sorted(manifest)[:10]}")
        sys.exit(1)

    clips = manifest[args.class_name]["clips"]
    print(f"Class    : {args.class_name} — {len(clips)} clips")
    print(f"Fills    : {args.fills}")
    print(f"Layers   : {'yes' if cfg['save_layers'] else 'no (--no-layers)'}")
    print(f"Attr maps: {'yes' if cfg['save_attribution'] else 'no (--no-attribution)'}")
    print(f"Masks    : {'yes' if cfg['save_masks'] else 'no (--no-masks)'}")

    # Verify label exists in model vocabulary before running any clips
    probe = np.zeros(stft_helper.sr, dtype=np.float32)
    Zxx_probe = stft_helper.compute_stft(probe)
    mel_probe  = stft_helper.to_logmel(Zxx_probe)
    probe_preds = model.infer_from_mel(mel_probe[np.newaxis])
    vocab = {p["label"] for p in probe_preds[0]}
    if args.class_name not in vocab:
        print(f"ERROR: '{args.class_name}' not in model vocabulary.")
        sys.exit(1)
    print(f"Label check passed.")

    model_slug = args.model_wrapper.replace(".py", "")
    class_slug = (args.class_name
                  .replace(", ", "_").replace(" ", "_")
                  .replace("(", "").replace(")", "").lower())

    base_dir = os.path.join(args.out_dir, model_slug, class_slug)
    out_dirs = {
        "embs":             os.path.join(base_dir, "embs"),
        "attribution_maps": os.path.join(base_dir, "attribution_maps"),
        "masks":            os.path.join(base_dir, "masks"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    summary_path = os.path.join(base_dir, "summary.csv")
    samples_path = os.path.join(base_dir, "samples.csv")
    errors_path  = os.path.join(base_dir, "errors.csv")

    # Resume: skip pairs already in summary.csv whose npz also exists (guards against CSV-before-flush crashes)
    completed = set()
    embs_dir = os.path.join(base_dir, "embs")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for row in csv.DictReader(f):
                cid, fill = row["clip_id"], row["fill"]
                npz = os.path.join(embs_dir, f"{cid}_perturbations_{fill}.npz")
                npy = os.path.join(embs_dir, f"{cid}_perturb_{fill}.npy")
                if (os.path.exists(npz) and os.path.getsize(npz) > 0
                        and os.path.exists(npy) and os.path.getsize(npy) > 0):
                    completed.add((cid, fill))

    summary_needs_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    samples_needs_header = not os.path.exists(samples_path) or os.path.getsize(samples_path) == 0
    errors_needs_header  = not os.path.exists(errors_path)  or os.path.getsize(errors_path)  == 0

    t_total = time.time()

    with open(summary_path, "a", newline="") as sf, \
         open(samples_path,  "a", newline="") as sp, \
         open(errors_path,   "a", newline="") as ef:

        sw = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        pw = csv.DictWriter(sp, fieldnames=SAMPLES_FIELDS)
        ew = csv.DictWriter(ef, fieldnames=["clip_id", "class_name", "model_id",
                                             "error_type", "message", "timestamp"])
        if summary_needs_header: sw.writeheader()
        if samples_needs_header: pw.writeheader()
        if errors_needs_header:  ew.writeheader()

        for i, clip in enumerate(clips):
            # Manifest entries can be a dict with a "filename" key or a bare path string.
            if isinstance(clip, dict):
                clip_path = os.path.join(args.wav_dir, clip.get("filename", clip.get("path", "")))
            else:
                clip_path = os.path.join(args.wav_dir, clip)

            clip_id    = os.path.splitext(os.path.basename(clip_path))[0]
            fills_todo = [f for f in args.fills if (clip_id, f) not in completed]

            if not fills_todo:
                print(f"[{i+1}/{len(clips)}] {clip_id} — all fills complete, skipping")
                continue
            print(f"\n[{i+1}/{len(clips)}] {clip_id}"
                  + (f" — resuming ({fills_todo})" if len(fills_todo) < len(args.fills) else ""))

            t_clip = time.time()
            try:
                process_clip(model, stft_helper, clip_path, out_dirs,
                             sw, pw, args.class_name, fills_todo, cfg)
                sf.flush(); sp.flush()
                print(f"  Clip done in {time.time() - t_clip:.0f}s")
            except Exception as e:
                print(f"  ERROR — {clip_id}: {e}")
                traceback.print_exc()
                ew.writerow({
                    "clip_id":    clip_id,
                    "class_name": args.class_name,
                    "model_id":   model.model_id,
                    "error_type": type(e).__name__,
                    "message":    str(e),
                    "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                ef.flush()

    print(f"\n=== {args.class_name} done in {(time.time() - t_total) / 3600:.1f}h ===")


if __name__ == "__main__":
    main()
