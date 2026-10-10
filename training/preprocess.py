"""DICOM Study -> fixed 44-slice uint8 stack, matching dreaddevelopment/knee-raptor-corpus.

Adapted from dreaddevelopment's 64-slice inference preprocessing. The 44-slice corpus uses the
older 15-85% slice span rather than that notebook's 6-94%; with that span this reproduces the
corpus stacks exactly (verify_preprocess.py: zero pixel difference on every slot).
"""
import glob

import numpy as np

IMG = 336
CROP_MM = 140.0
SPAN = (0.15, 0.85)
PCT = (2.0, 98.0)
# (plane, preferred Fluid_Sensitive or -1 for no preference, slices)
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
         ("Coronal", 0, 6), ("Axial", -1, 8)]
MAXS = sum(s[2] for s in SLOTS)  # 44


def _order_and_meta(sdir):
    import pydicom
    recs, spacings = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            spacings.append(ps)
            recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    return [(f, ps) for _, f, ps in recs], (float(np.median(spacings)) if spacings else 0.5)


def _read_px(f):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def _crop_resize(a, ps):
    import cv2
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def _pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(study_dir, series_rows):
    """series_rows: list of dicts (train_series/test_series rows) for this Study, in CSV order."""
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in SLOTS:
        start = idx
        r = _pick_series(series_rows, plane, fluid, used)
        files = []
        if r is not None:
            used.add(r["SeriesInstanceUID"])
            files, med_ps = _order_and_meta(f"{study_dir}/{r['SeriesInstanceUID']}")
        if files:
            n = len(files)
            lo, hi = int(n * SPAN[0]), max(int(n * SPAN[1]) - 1, int(n * SPAN[0]))
            picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
            arrs, pss = [], []
            for p in picks:
                fp, ps = files[min(p, n - 1)]
                try:
                    arrs.append(_read_px(fp)); pss.append(ps)
                except Exception:
                    arrs.append(None); pss.append(med_ps)
            valid = [a for a in arrs if a is not None]
            loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), PCT)
                        if valid else (0.0, 1.0))
            for a, ps in zip(arrs, pss):
                if a is not None:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    vol[idx] = (_crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
                idx += 1
        idx = start + k
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask
