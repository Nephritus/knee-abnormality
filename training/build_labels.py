"""Build the Report Label file and the fold file that the teacher runs train on.

Merges two public Report Label sets by averaging their probabilities, assigns every
non-gold training Study to one of N folds (Studies sharing an identical Report stay in
the same fold), and prints macro AUC against the Gold Labels for each input and the merge.

    python training/build_labels.py            # writes into build/kit/
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "kit"
N_FOLDS = 3
SEED = 42

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def nartaa(path):
    d = pd.read_parquet(path).set_index("StudyInstanceUID")
    return pd.DataFrame({c: d["y_" + c.replace("'", "").replace(" ", "_")] for c in LAB})


def gold_auc(labels, gold):
    ids = gold.index.intersection(labels.index)
    aucs = [roc_auc_score(gold.loc[ids, c], labels.loc[ids, c].astype(float)) for c in LAB]
    return float(np.mean(aucs)), aucs


def main():
    train = pd.read_csv(DATA / "train.csv")
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    gold = train[train[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")[LAB]

    sources = {
        "stevenleehans_v4": pd.read_csv(
            DATA / "labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
        ).set_index("StudyInstanceUID")[LAB],
        "nartaa_v1": nartaa(DATA / "labels/rsna-knee-hpo-assets/train_labels_v1.parquet"),
    }
    ids = train["StudyInstanceUID"]
    merged = sum(s.reindex(ids).astype(float) for s in sources.values()) / len(sources)
    merged = merged.fillna(0.5).clip(0, 1)

    for name, s in {**sources, "merged": merged}.items():
        macro, _ = gold_auc(s, gold)
        print(f"{name:18s} gold macro AUC {macro:.4f}")

    # Folds over non-gold Studies, grouped by Report text so duplicate Reports never straddle folds.
    rng = np.random.default_rng(SEED)
    pool = train[~train["StudyInstanceUID"].isin(gold.index)]
    groups = pool.groupby(pool["Report"].fillna("").map(
        lambda r: hashlib.sha1(r.encode()).hexdigest()))["StudyInstanceUID"].apply(list).tolist()
    rng.shuffle(groups)
    sizes = [0] * N_FOLDS
    folds = {}
    for g in sorted(groups, key=len, reverse=True):
        f = int(np.argmin(sizes))
        for uid in g:
            folds[uid] = f
        sizes[f] += len(g)
    print(f"folds {sizes} from {len(groups)} report groups ({len(pool)} studies)")

    OUT.mkdir(parents=True, exist_ok=True)
    merged.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_parquet(
        OUT / "labels_report.parquet", index=False)
    json.dump({"folds": folds, "n_folds": N_FOLDS, "seed": SEED},
              open(OUT / "folds.json", "w"))
    print(f"wrote {OUT / 'labels_report.parquet'} and {OUT / 'folds.json'}")


if __name__ == "__main__":
    main()
