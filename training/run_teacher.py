"""Kaggle launcher for training runs: optional smoke test, then one job per GPU in parallel.

Expects the kit dataset (train_knee.py, label parquets, folds.json) and both
knee-raptor-corpus parts attached. Outputs land in /kaggle/working.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# One job per GPU. fold=None trains on every non-gold Study (no held-out predictions).
# History: session 1 = teacher folds 0-1; session 2 = teacher fold 2 + report_full.
JOBS = [
    {"tag": "student_s42", "fold": None, "labels": "labels_student.parquet", "seed": 42},
    {"tag": "student_s7", "fold": None, "labels": "labels_student.parquet", "seed": 7},
]
SMOKE = False
N_FOLDS = 3
EPOCHS = 12
RES = 224
ARCH_CANDIDATES = [
    "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
    "coatnet_2_rw_224.sw_in12k_ft_in1k",
    "coatnet_rmlp_1_rw_224.sw_in1k",
]

WORK = Path("/kaggle/working")


def find_kit():
    for p in Path("/kaggle/input").rglob("train_knee.py"):
        return p.parent
    sys.exit("kit dataset with train_knee.py is not attached")


def pick_arch():
    import timm
    available = set(timm.list_pretrained("coatnet*"))
    for a in ARCH_CANDIDATES:
        if a in available:
            return a
    sys.exit(f"none of {ARCH_CANDIDATES} in timm {timm.__version__}: {sorted(available)}")


def cmd(arch, kit, job, extra=()):
    folds = ([] if job["fold"] is None else
             ["--folds", str(N_FOLDS), "--fold", str(job["fold"]), "--fold_file", str(kit / "folds.json")])
    return [sys.executable, str(WORK / "train_knee.py"), "--arch", arch, "--res", str(RES),
            "--epochs", str(EPOCHS), "--bs", "8", "--k", "12", "--k_eval", "24", "--grad_ckpt",
            "--workers", "2", "--labels", str(kit / job["labels"]), *folds, "--tag", job["tag"],
            "--seed", str(job.get("seed", 42)), *extra]


def main():
    kit = find_kit()
    shutil.copy(kit / "train_knee.py", WORK / "train_knee.py")
    arch = pick_arch()
    print(f"arch {arch} | kit {kit} | jobs {[j['tag'] for j in JOBS]}", flush=True)

    t0 = time.time()
    if SMOKE:
        smoke = subprocess.run(cmd(arch, kit, {**JOBS[0], "tag": "smoke"}, ["--smoke"]))
        print(f"smoke exit {smoke.returncode} in {time.time() - t0:.0f}s", flush=True)
        if smoke.returncode != 0:
            sys.exit("smoke test failed; not starting the real runs")
        for f in WORK.glob("*smoke*"):
            f.unlink()

    import torch
    n_gpu = max(torch.cuda.device_count(), 1)
    print(f"{n_gpu} GPU(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
          flush=True)
    procs = []
    for i, job in enumerate(JOBS):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i % n_gpu)}
        log = open(WORK / f"{job['tag']}.log", "w")
        procs.append((job["tag"], log, subprocess.Popen(cmd(arch, kit, job), env=env,
                                                        stdout=log, stderr=subprocess.STDOUT)))
    while any(p.poll() is None for _, _, p in procs):
        time.sleep(300)
        for tag, _, _ in procs:
            lines = (WORK / f"{tag}.log").read_text().splitlines()
            eps = [l for l in lines if l.startswith("ep")]
            print(f"[{(time.time() - t0) / 3600:.2f}h] {tag}: {eps[-1] if eps else (lines[-1] if lines else '-')}",
                  flush=True)
    for tag, log, p in procs:
        log.close()
        print(f"{tag} exit {p.returncode}", flush=True)
        print((WORK / f"{tag}.log").read_text()[-2500:], flush=True)
    if any(p.returncode for _, _, p in procs):
        sys.exit(1)


if __name__ == "__main__":
    main()
