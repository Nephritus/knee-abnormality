"""Score test Studies with trained checkpoints from train_knee.py.

DICOM -> preprocess.build_study (corpus-exact) -> the trainer's evaluation windowing ->
RaptorClassifier. Checkpoints are averaged in probability space. Writes a submission-format CSV.

    from student_infer import predict
    predict(comp_root, [ckpt, ...], "/kaggle/working/_student.csv", deadline=time.time() + 3600)
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import preprocess as pp
from train_knee import LAB, RaptorClassifier, build_backbone

K_EVAL = 24  # matches the trainer's --k_eval


def eval_windows(vol, mask, k, res):
    """Same as train_knee.StudyWindows with train=False, roi off, norm 'none'."""
    D = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    cs = [cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)]
    tiles = []
    for c in cs:
        c = max(1, min(c, D - 2))
        t = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        tiles.append(t)
    return torch.stack(tiles)


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bb = build_backbone(ck["arch"], pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), int(ck["res"])


@torch.no_grad()
def predict(comp_root, ckpt_paths, out_csv, deadline=None, workers=4):
    comp = Path(comp_root)
    test = pd.read_csv(comp / "test.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(comp / "test_series.csv", dtype={"StudyInstanceUID": str})
    rows = {s: g.to_dict("records") for s, g in series.groupby("StudyInstanceUID", sort=False)}
    uids = test["StudyInstanceUID"].tolist()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load(p, dev) for p in ckpt_paths]
    print(f"[student] {len(models)} checkpoint(s), {len(uids)} studies, device {dev}", flush=True)

    def prep(uid):
        return pp.build_study(str(comp / "test_series" / uid), rows.get(uid, []))

    preds = np.full((len(uids), len(LAB)), np.nan, np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for i, (vol, mask) in enumerate(pool.map(prep, uids)):
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"student inference passed its deadline after {i} studies")
            ps = []
            for model, res in models:
                x = eval_windows(vol, mask, K_EVAL, res)[None].to(dev)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    ps.append(torch.sigmoid(model(x).float())[0].cpu().numpy())
            preds[i] = np.mean(ps, 0)
            if i % 100 == 0:
                print(f"[student] {i}/{len(uids)} in {time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(preds).all(), "student produced non-finite predictions"
    out = pd.DataFrame(preds, columns=LAB)
    out.insert(0, "StudyInstanceUID", uids)
    out.to_csv(out_csv, index=False)
    print(f"[student] wrote {out_csv} in {time.time() - t0:.0f}s", flush=True)
    return out
