"""Inference windowing must match the trainer's evaluation windowing exactly."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
tk = pytest.importorskip("train_knee")
import student_infer as si  # noqa: E402


@pytest.mark.parametrize("seed", range(4))
def test_eval_windows_match_trainer(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    vol = rng.integers(0, 255, (44, 336, 336), dtype=np.uint8)
    mask = np.ones(44, np.uint8)
    mask[rng.integers(0, 44, 6)] = 0
    if seed == 1:  # mostly empty study
        mask[:] = 0
        mask[10:20] = 1
    monkeypatch.setattr(tk, "_open_corpus", lambda: (vol[None], mask[None]))
    ds = tk.StudyWindows(".", ["u"], {"u": 0}, {"u": np.zeros(12, np.float32)}, 224, si.K_EVAL, train=False)
    expected, _ = ds[0]
    assert (si.eval_windows(vol, mask, si.K_EVAL, 224) - expected).abs().max().item() == 0.0
