# Knee Abnormality Detection

Predicting, for each knee MRI Study, the probability of twelve Abnormalities, for the RSNA 2026 Knee Abnormality Detection Kaggle competition. Scored by macro-averaged AUC-ROC.

## Language

**Study**:
One knee MRI exam, identified by `StudyInstanceUID`. One Study is one row of a submission.
_Avoid_: Exam, case, patient

**Series**:
One MRI sequence within a Study, identified by `SeriesInstanceUID`; a stack of DICOM slices in a single Anatomical Plane.
_Avoid_: Scan, sequence, volume

**Anatomical Plane**:
The orientation of a Series: Axial, Sagittal or Coronal.
_Avoid_: View, orientation

**Abnormality**:
One of the twelve clinically important conditions predicted per Study (ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture); each is a column in the submission.
_Avoid_: Finding, label, class

**Gold Label**:
An expert radiologist's reading of an Abnormality for a Study; only a few dozen training Studies carry them.
_Avoid_: Ground truth, official label

**Report Label**:
An Abnormality target derived by reading a Study's Report (by rules or a language model); noisier than a Gold Label.
_Avoid_: LLM label, pseudo-label, weak label

**Unaddressed**:
A Report that says nothing about a given Abnormality; its meaning differs per Abnormality (near-certain absence for some, uninformative for others).
_Avoid_: Missing, silent, unknown

**Teacher Label**:
An Abnormality target taken from a teacher model's held-out prediction for a training Study, used (often mixed with Report Labels) to train a Student.
_Avoid_: Pseudo-label, soft label, OOF label

**Student**:
The model trained on Teacher Labels whose predictions are submitted.
_Avoid_: Final model, distilled model

**Community Stack**:
The shared public ensemble of pretrained readers (DINO, RadImageNet, Raptor, CoAtNet) that most public notebooks run unchanged, scoring about 0.943.
_Avoid_: Public baseline, the blend

**Report**:
The free-text radiology report attached to a training Study, in one of about a dozen languages.
_Avoid_: Note, text
