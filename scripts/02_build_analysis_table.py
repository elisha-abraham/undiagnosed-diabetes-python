"""Merge the NHANES files into one analysis table.

Reads   data/raw/*.xpt
Writes  data/analysis/nhanes_participants.csv   one row per participant
        results/sample_flow.csv                 how the analysis sample was reached

    python scripts/02_build_analysis_table.py

Every participant is kept, including those outside the analysis, and
marked with `in_analysis`. The survey's variance formula needs the whole
design (all 30 sampling units), so the sample is never cut down by
deleting rows; it is narrowed with a flag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "analysis"
RESULTS = ROOT / "results"

RACE = {1: "Hispanic", 2: "Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian", 7: "Other or multiracial"}
EDUCATION = {1: "Less than high school", 2: "Less than high school", 3: "High school",
             4: "Some college", 5: "College graduate"}


def read(name: str) -> pd.DataFrame:
    """Read a SAS transport file. pandas returns exact zeros as 5.4e-79; put them back.

    mask() is used, not where(): a missing value fails every comparison, so
    where(abs > tiny) would quietly turn every missing value into zero.
    """
    table = pd.read_sas(RAW / f"{name}.xpt", format="xport")
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].mask(table[numeric].abs() < 1e-70, 0.0)
    table["SEQN"] = table["SEQN"].astype("int64")
    return table


def yes_no(series: pd.Series, yes: tuple = (1,), no: tuple = (2,)) -> pd.Series:
    """Coded answer -> 1.0 / 0.0, with refused, don't know and blank left missing."""
    return pd.Series(np.select([series.isin(yes), series.isin(no)], [1.0, 0.0], default=np.nan), index=series.index)


def main() -> None:
    demo = read("DEMO_L")
    table = demo[["SEQN", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "RIDEXPRG", "SDMVSTRA", "SDMVPSU"]]
    for name, columns in {
        "DIQ_L": ["DIQ010", "DIQ160"],
        "GHB_L": ["WTPH2YR", "LBXGH"],
        "BMX_L": ["BMXBMI"],
        "HIQ_L": ["HIQ011"],
        "HUQ_L": ["HUQ030"],
        "BPQ_L": ["BPQ020"],
    }.items():
        table = table.merge(read(name)[["SEQN"] + columns], on="SEQN", how="left", validate="one_to_one")

    out = pd.DataFrame({
        "seqn": table.SEQN,
        "stratum": table.SDMVSTRA.astype(int),
        "psu": table.SDMVPSU.astype(int),
        "lab_weight": table.WTPH2YR.fillna(0.0),
        "age": table.RIDAGEYR,
        "sex": table.RIAGENDR.map({1: "Male", 2: "Female"}),
        "race_ethnicity": table.RIDRETH3.map(RACE),
        "education": table.DMDEDUC2.map(EDUCATION),
        "pregnant": (table.RIDEXPRG == 1).astype(int),
        "bmi": table.BMXBMI,
        "hba1c": table.LBXGH,
        "diagnosed": yes_no(table.DIQ010, yes=(1,), no=(2, 3)),       # "borderline" is not a diagnosis
        "told_prediabetes": yes_no(table.DIQ160),
        "insured": yes_no(table.HIQ011),
        "usual_place_of_care": yes_no(table.HUQ030, yes=(1, 3), no=(2,)),
        "told_high_blood_pressure": yes_no(table.BPQ020),
    })
    out["age_group"] = pd.cut(out.age, [19, 39, 59, 200], labels=["20-39", "40-59", "60 and over"]).astype(object)
    out["bmi_group"] = pd.cut(out.bmi, [0, 24.999, 29.999, 200], labels=["Under 25", "25 to 29.9", "30 and over"]).astype(object)

    # Glycaemic status: a diagnosis comes first; otherwise the blood test decides.
    status = np.select(
        [out.diagnosed == 1, out.hba1c >= 6.5, out.hba1c >= 5.7, out.hba1c < 5.7],
        ["Diagnosed diabetes", "Undiagnosed diabetes", "Prediabetes range", "Normal range"], default=None)
    out["status"] = pd.Series(status, index=out.index).where(out.diagnosed.notna() & out.hba1c.notna())
    out["undiagnosed"] = np.where(out.status.notna(), (out.status == "Undiagnosed diabetes").astype(float), np.nan)
    out["has_diabetes"] = np.where(out.status.notna(), out.status.isin(["Diagnosed diabetes", "Undiagnosed diabetes"]).astype(float), np.nan)

    # ---- analysis sample, one exclusion at a time ----
    steps = [
        ("All participants", pd.Series(True, index=out.index)),
        ("Aged 20 or over", out.age >= 20),
        ("Not pregnant", out.pregnant == 0),
        ("Gave a blood sample (lab weight above zero)", out.lab_weight > 0),
        ("HbA1c result available", out.hba1c.notna()),
        ("Answered the diabetes question", out.diagnosed.notna()),
    ]
    keep = pd.Series(True, index=out.index)
    flow = []
    for label, condition in steps:
        keep &= condition
        flow.append({"step": label, "participants": int(keep.sum())})
    out["in_analysis"] = keep.astype(int)

    # ---- checks ----
    assert out.seqn.is_unique, "a participant appears twice"
    design = out.groupby("stratum").psu.nunique()
    assert len(design) == 15 and (design == 2).all(), "expected 15 strata with 2 sampling units each"
    assert out.loc[keep, "hba1c"].between(3, 20).all(), "HbA1c outside a plausible range"
    assert out.loc[keep, "status"].notna().all()
    assert (out.loc[keep, "lab_weight"] > 0).all()

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(OUT / "nhanes_participants.csv", index=False)
    pd.DataFrame(flow).to_csv(RESULTS / "sample_flow.csv", index=False)
    print(pd.DataFrame(flow).to_string(index=False))
    print("\nunweighted status counts in the analysis sample:")
    print(out.loc[keep, "status"].value_counts().to_string())
    print("\nmissing among the analysis sample:", out.loc[keep, ["bmi", "education", "insured", "usual_place_of_care"]].isna().sum().to_dict())


if __name__ == "__main__":
    main()
