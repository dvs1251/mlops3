"""
Generates a PLACEHOLDER kidney-disease dataset that matches the schema of the
UCI "Chronic Kidney Disease" dataset (25 columns, ~400 rows) so the rest of
the Git+DVC pipeline can be built and exercised right away.

This is NOT the real UCI data (no internet access was available while
building this project). Replace data/cleaned_kidney_disease.csv with the
real file (same column names) as soon as you have it -- nothing else in the
pipeline needs to change.

Real dataset: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 400

age = rng.normal(51, 17, N).clip(2, 90).round(0)
bp = rng.choice([50, 60, 70, 76, 80, 90, 100, 110, 120, 140, 150, 160, 180], N,
                p=None)
sg = rng.choice([1.005, 1.010, 1.015, 1.020, 1.025], N,
                 p=[0.15, 0.2, 0.25, 0.25, 0.15])
al = rng.choice([0, 1, 2, 3, 4, 5], N, p=[0.55, 0.12, 0.12, 0.11, 0.06, 0.04])
su = rng.choice([0, 1, 2, 3, 4, 5], N, p=[0.72, 0.08, 0.06, 0.06, 0.04, 0.04])
rbc = rng.choice(["normal", "abnormal"], N, p=[0.8, 0.2])
pc = rng.choice(["normal", "abnormal"], N, p=[0.75, 0.25])
pcc = rng.choice(["present", "notpresent"], N, p=[0.15, 0.85])
ba = rng.choice(["present", "notpresent"], N, p=[0.1, 0.9])
bgr = rng.normal(140, 55, N).clip(60, 490).round(0)
bu = rng.normal(55, 45, N).clip(10, 390).round(0)
sc = rng.normal(2.8, 3.0, N).clip(0.4, 25).round(1)
sod = rng.normal(137, 9, N).clip(105, 165).round(0)
pot = rng.normal(4.5, 1.4, N).clip(2.0, 12.0).round(1)
hemo = rng.normal(12.6, 3.0, N).clip(3.1, 17.8).round(1)
pcv = rng.normal(38, 9, N).clip(9, 54).round(0)
wc = rng.normal(8400, 2700, N).clip(2200, 26400).round(0)
rc = rng.normal(4.7, 1.0, N).clip(2.1, 8.0).round(1)
htn = rng.choice(["yes", "no"], N, p=[0.37, 0.63])
dm = rng.choice(["yes", "no"], N, p=[0.34, 0.66])
cad = rng.choice(["yes", "no"], N, p=[0.09, 0.91])
appet = rng.choice(["good", "poor"], N, p=[0.78, 0.22])
pe = rng.choice(["yes", "no"], N, p=[0.19, 0.81])
ane = rng.choice(["yes", "no"], N, p=[0.15, 0.85])

# Build a plausible CKD label from the clinical signals rather than pure noise
risk = (
    (al >= 1).astype(int)
    + (su >= 1).astype(int)
    + (sc > 1.3).astype(int)
    + (hemo < 11).astype(int)
    + (htn == "yes").astype(int)
    + (dm == "yes").astype(int)
    + (rbc == "abnormal").astype(int)
    + (bu > 45).astype(int)
)
classification = np.where(risk >= 3, "ckd", "notckd")

df = pd.DataFrame({
    "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
    "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
    "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
    "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
    "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane,
    "classification": classification,
})

# sprinkle missing values on 5 distinct rows only, so dropna() leaves exactly
# 395 rows (matching Dataset V1 in the report) while the imputed version
# keeps all 400 (matching Dataset V2)
missing_rows = rng.choice(N, size=5, replace=False)
missing_cols = rng.choice(
    ["age", "bp", "sg", "rbc", "hemo", "wc"], size=5, replace=True
)
for r, c in zip(missing_rows, missing_cols):
    df.loc[r, c] = np.nan

import os
OUT_PATH = os.path.join(os.path.dirname(__file__), "cleaned_kidney_disease_v2_raw.csv")
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows -> {OUT_PATH}")
print(df["classification"].value_counts())
