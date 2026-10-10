"""Build the Student's targets: Teacher Labels mixed 50/50 with Report Labels.

Reads every teacher fold's held-out predictions (epoch-ensemble files) from build/teacher/,
checks each non-gold Study is covered exactly once, and writes build/kit/labels_student.parquet.

    python training/build_student_labels.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT / "build" / "teacher"
KIT = ROOT / "build" / "kit"
TEACHER_WEIGHT = 0.5

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    report = pd.read_parquet(KIT / "labels_report.parquet").set_index("StudyInstanceUID")[LAB]
    parts = []
    for f in sorted(TEACHER.glob("raptor_oof_teacher*_epens_fold*.npz")):
        z = np.load(f, allow_pickle=True)
        parts.append(pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB))
        print(f"{f.name}: {len(z['ids'])} studies")
    teacher = pd.concat(parts)
    assert teacher.index.is_unique, "a Study appears in more than one fold"
    missing = report.index.difference(teacher.index)
    print(f"teacher covers {len(teacher)} studies; {len(missing)} without a Teacher Label "
          "(expected: the 58 gold Studies)")

    student = report.copy()
    ix = teacher.index
    student.loc[ix] = TEACHER_WEIGHT * teacher.loc[ix, LAB] + (1 - TEACHER_WEIGHT) * report.loc[ix]
    student.clip(0, 1).reset_index().to_parquet(KIT / "labels_student.parquet", index=False)
    print(f"wrote {KIT / 'labels_student.parquet'}")


if __name__ == "__main__":
    main()
