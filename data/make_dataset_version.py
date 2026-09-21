"""
Builds data/cleaned_kidney_disease.csv from the raw placeholder data.

Usage:
    python data/make_dataset_version.py v1   # 395 rows, rows with any NaN dropped
    python data/make_dataset_version.py v2   # 400 rows, missing values imputed

Run v1 first, `dvc add` + `git commit` it, then run v2 and `dvc add` +
`git commit` again -- this reproduces the "Dataset V1 -> Dataset V2" step
from the report (Sections 3-4) on the kidney-disease data.
"""
import sys

import pandas as pd

RAW = "data/cleaned_kidney_disease_v2_raw.csv"
OUT = "data/cleaned_kidney_disease.csv"


def main(version: str):
    df = pd.read_csv(RAW)

    if version == "v1":
        df = df.dropna().reset_index(drop=True)
    elif version == "v2":
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(exclude="number").columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    else:
        raise SystemExit("version must be 'v1' or 'v2'")

    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT} ({version})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v2")
