"""Submission: Student checkpoints only (no Community Stack)."""
import sys
import time
from pathlib import Path

KIT = next(Path("/kaggle/input").rglob("student_infer.py")).parent
sys.path.insert(0, str(KIT))
from student_infer import predict  # noqa: E402

T0 = time.time()
comp = next(p.parent for p in Path("/kaggle/input").rglob("test.csv") if (p.parent / "test_series").is_dir())
ckpts = sorted(Path("/kaggle/input").rglob("raptor_ft_student_*_swa.pt"))
assert ckpts, "no Student checkpoints attached"
print("checkpoints:", [c.name for c in ckpts], flush=True)
predict(comp, ckpts, "/kaggle/working/submission.csv", deadline=T0 + 8.5 * 3600)
