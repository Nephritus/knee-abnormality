"""CPU Kaggle kernel: rebuild training Studies from DICOM and compare with the corpus stacks.

Prints per-slot mean absolute pixel difference and correlation, plus mask agreement.
A faithful reimplementation should show near-zero difference on every slot.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = next(Path("/kaggle/input").rglob("preprocess.py")).parent
sys.path.insert(0, str(KIT))
import preprocess as pp  # noqa: E402


def find(name):
    return next(Path("/kaggle/input").rglob(name))


# (SPAN, PCT) settings to compare; the first is preprocess.py's default. A wider grid found
# (0.15, 0.85) / (2, 98) to be the only exact match.
GRID = [((0.15, 0.85), (2.0, 98.0))]


def main(n_studies=6, dump=0):
    comp = find("train_series.csv").parent
    ids = np.concatenate([np.load(find("all_ids.npy"), allow_pickle=True).astype(str),
                          np.load(find("extra_ids.npy"), allow_pickle=True).astype(str)])
    vols = [np.load(find("all_vols.npy"), mmap_mode="r"), np.load(find("extra_vols.npy"), mmap_mode="r")]
    masks = np.concatenate([np.load(find("all_masks.npy")), np.load(find("extra_masks.npy"))])
    na = vols[0].shape[0]
    print("corpus", len(ids), vols[0].shape, "| ours", (pp.MAXS, pp.IMG, pp.IMG), flush=True)

    series = pd.read_csv(comp / "train_series.csv")
    rows_by_study = {s: g.to_dict("records") for s, g in series.groupby("StudyInstanceUID", sort=False)}
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ids), n_studies, replace=False)
    for span, pct in GRID:
        pp.SPAN, pp.PCT = span, pct
        print(f"\n=== SPAN {span} PCT {pct}", flush=True)
        run(pp, pick, ids, vols, masks, na, comp, rows_by_study, dump)


def run(pp, pick, ids, vols, masks, na, comp, rows_by_study, dump):
    bounds = np.cumsum([0] + [s[2] for s in pp.SLOTS])
    diffs, corrs, mask_eq = [], [], []
    for row in pick:
        uid = ids[row]
        ref = np.asarray(vols[0][row] if row < na else vols[1][row - na]).astype(np.float32)
        vol, m = pp.build_study(str(comp / "train_series" / uid), rows_by_study.get(uid, []))
        vol = vol.astype(np.float32)
        d, c = [], []
        for s in range(len(pp.SLOTS)):
            a, b = vol[bounds[s]:bounds[s + 1]], ref[bounds[s]:bounds[s + 1]]
            d.append(float(np.abs(a - b).mean()))
            c.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan)
        diffs.append(d); corrs.append(c); mask_eq.append(float((m == masks[row]).mean()))
        if len(diffs) <= dump:  # raw pair for offline diagnosis
            np.savez_compressed(f"/kaggle/working/pair_{len(diffs)}.npz", ours=vol.astype(np.uint8),
                                ref=ref.astype(np.uint8), uid=uid)
        print(f"{uid[-12:]} MAE/slot {np.round(d, 2).tolist()} corr {np.round(c, 3).tolist()} "
              f"mask {mask_eq[-1]:.2f}", flush=True)
    print("MEAN MAE per slot", np.round(np.nanmean(diffs, 0), 3).tolist())
    print("MEAN corr per slot", np.round(np.nanmean(corrs, 0), 4).tolist())
    print("mask agreement", round(float(np.mean(mask_eq)), 4))


if __name__ == "__main__":
    main()
