# %% [markdown]
# # Undiagnosed diabetes in US adults
#
# **How many adults have diabetes without knowing it, who are they, and which screening rule finds them?**
#
# Data: NHANES August 2021 to August 2023 (US National Center for Health Statistics). NHANES examines a
# sample of the population and takes blood, so it can find diabetes in people who have never been told
# they have it.
#
# Definitions used throughout:
#
# - **Diagnosed diabetes**: answered yes to "has a doctor ever told you that you have diabetes?"
# - **Undiagnosed diabetes**: no such diagnosis, and HbA1c of 6.5% or higher in the survey's blood test.
# - **Prediabetes range**: no diagnosis, HbA1c from 5.7% to 6.4%.
#
# This notebook is generated from `analysis.py` by `scripts/build_notebook.py`.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "survey.py").exists())
sys.path.insert(0, str(ROOT / "src"))
import charts  # noqa: E402
from survey import Design, dummies  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
pd.set_option("display.width", 160, "display.max_columns", 20, "display.float_format", "{:,.2f}".format)

data = pd.read_csv(ROOT / "data" / "analysis" / "nhanes_participants.csv")
data["insurance"] = data.insured.map({1.0: "Insured", 0.0: "Uninsured"})
data["usual_care"] = data.usual_place_of_care.map({1.0: "Has a usual place of care", 0.0: "No usual place of care"})

# The blood-test weight, with the survey's strata and sampling units.
design = Design(data, weight="lab_weight", stratum="stratum", psu="psu")
adults = data.in_analysis == 1
print(f"{len(data):,} participants; {design.n_strata} strata, {design.n_psu} sampling units, "
      f"{design.df} degrees of freedom; t multiplier {design.t:.3f}")

# %% [markdown]
# ## 1. Who is in the analysis
#
# Adults aged 20 or over who were not pregnant, gave blood, have an HbA1c result and answered the
# diabetes question. People outside this group stay in the data with a flag: the variance formula
# needs every sampling unit, so nobody is deleted.

# %%
display(pd.read_csv(RESULTS / "sample_flow.csv"))

# %% [markdown]
# ## 2. How common is it?
#
# Percentages are weighted to the US adult population. Intervals are 95% and account for the survey design.


# %%
def percent(result: dict) -> dict:
    """An estimate and its interval, as percentages."""
    return {"estimate": 100 * result["estimate"], "low": 100 * result["low"], "high": 100 * result["high"], "n": result["n"]}


STATUSES = ["Normal range", "Prediabetes range", "Undiagnosed diabetes", "Diagnosed diabetes"]
status_rows = []
for status in STATUSES:
    indicator = (data.status == status).astype(float).where(data.status.notna())
    people = design.total(indicator, adults)
    status_rows.append({"status": status, **percent(design.proportion(indicator, adults)),
                        "million_adults": people["estimate"] / 1e6,
                        "million_low": people["low"] / 1e6, "million_high": people["high"] / 1e6})
status_table = pd.DataFrame(status_rows)
status_table.to_csv(RESULTS / "glycaemic_status.csv", index=False)
display(status_table)

with_diabetes = adults & (data.has_diabetes == 1)
unaware = percent(design.proportion(data.undiagnosed, with_diabetes))
print(f"Of adults with diabetes, {unaware['estimate']:.1f}% ({unaware['low']:.1f} to {unaware['high']:.1f}) are undiagnosed "
      f"(n = {unaware['n']:,} with diabetes).")

# %%
path = charts.interval_bars(
    [{"label": row.status, "estimate": row.estimate, "low": row.low, "high": row.high,
      "note": f"{row.million_adults:.0f} million" if row.million_adults >= 9.5 else f"{row.million_adults:.1f} million"}
     for row in status_table.itertuples()],
    "One adult in five is in the prediabetes range; 5.5 million have undiagnosed diabetes",
    "Blood-test status of US adults aged 20 and over, with 95% intervals and the number of people.",
    FIGURES, "glycaemic_status", x_max=96, ticks=[0, 20, 40, 60], label_width=0.30)
display(Image(str(path), width=720))

# %% [markdown]
# ## 3. Among people with diabetes, who does not know?
#
# The share of diabetes that is undiagnosed, by group. Groups with fewer than 30 people with diabetes
# in the sample are not reported.

# %%
GROUPS = {
    "Age": ("age_group", ["20-39", "40-59", "60 and over"]),
    "Sex": ("sex", ["Female", "Male"]),
    "Race and ethnicity": ("race_ethnicity", ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian", "Other or multiracial"]),
    "Health insurance": ("insurance", ["Insured", "Uninsured"]),
    "Usual place of care": ("usual_care", ["Has a usual place of care", "No usual place of care"]),
}
share_rows = [{"group": "All adults with diabetes", "level": "All", **unaware}]
for group, (variable, levels) in GROUPS.items():
    for level in levels:
        share_rows.append({"group": group, "level": level,
                           **percent(design.proportion(data.undiagnosed, with_diabetes & (data[variable] == level)))})
share_table = pd.DataFrame(share_rows)
share_table["reported"] = share_table.n >= 30
share_table.to_csv(RESULTS / "undiagnosed_share_by_group.csv", index=False)
display(share_table)

# %%
chart_rows = [{"label": "All adults with diabetes", **unaware, "note": ""}]
for group in ["Age", "Health insurance", "Usual place of care"]:
    chart_rows.append({"label": group, "estimate": None})
    for row in share_table[(share_table.group == group) & share_table.reported].itertuples():
        chart_rows.append({"label": row.level, "estimate": row.estimate, "low": row.low, "high": row.high})
path = charts.interval_bars(
    chart_rows,
    "One in six adults with diabetes has not been told; more where care is out of reach",
    "Share of diabetes that is undiagnosed (no diagnosis, HbA1c 6.5% or higher), with 95% intervals.",
    FIGURES, "undiagnosed_share_by_group", x_max=62, ticks=[0, 10, 20, 30, 40, 50])
display(Image(str(path), width=720))

# %% [markdown]
# ## 4. Among adults with no diagnosis, what predicts an HbA1c of 6.5% or higher?
#
# A logistic regression, weighted, with standard errors from the survey design. With 15 degrees of
# freedom the model has to stay small: five predictors, ten coefficients.

# %%
no_diagnosis = adults & (data.diagnosed == 0)
LEVELS = {
    "age_group": ["20-39", "40-59", "60 and over"],
    "sex": ["Female", "Male"],
    "race_ethnicity": ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian", "Other or multiracial"],
    "bmi_group": ["Under 25", "25 to 29.9", "30 and over"],
    "insurance": ["Insured", "Uninsured"],
}
X = dummies(data, LEVELS)
model = design.logistic(X, data.undiagnosed, no_diagnosis)
model.to_csv(RESULTS / "model_odds_ratios.csv", index=False)
display(model[["term", "odds_ratio", "low", "high", "p_value"]])
print(f"n = {model.n.iloc[0]:,} adults without a diagnosis; {model.events.iloc[0]:,} with HbA1c of 6.5% or higher")

# Crude (unadjusted) prevalence of undiagnosed diabetes in the same groups, for context.
crude_rows = []
for variable, levels in LEVELS.items():
    for level in levels:
        crude_rows.append({"variable": variable, "level": level,
                           **percent(design.proportion(data.undiagnosed, no_diagnosis & (data[variable] == level)))})
crude = pd.DataFrame(crude_rows)
crude.to_csv(RESULTS / "undiagnosed_prevalence_by_group.csv", index=False)
display(crude)

# %% [markdown]
# **Check.** The coefficients are fitted here with Newton's method. statsmodels, fitting the same
# weighted model, must give the same coefficients.

# %%
import statsmodels.api as sm  # noqa: E402

rows = no_diagnosis & X.notna().all(axis=1) & data.undiagnosed.notna()
reference = sm.GLM(data.undiagnosed[rows], X[rows], family=sm.families.Binomial(),
                   freq_weights=data.lab_weight[rows] / data.lab_weight[rows].mean()).fit()
largest_gap = float(np.abs(reference.params.to_numpy() - model.coefficient.to_numpy()).max())
assert largest_gap < 1e-6, largest_gap
print(f"Largest difference from statsmodels across {len(model)} coefficients: {largest_gap:.1e}")

# %%
LABELS = {"age_group": "Age", "sex": "Sex", "race_ethnicity": "Race and ethnicity", "bmi_group": "Body mass index", "insurance": "Health insurance"}
by_term = model.set_index("term")
forest_rows = []
for variable, levels in LEVELS.items():
    forest_rows.append({"label": LABELS[variable]})
    forest_rows.append({"label": levels[0], "reference": True})
    for level in levels[1:]:
        row = by_term.loc[f"{variable}: {level}"]
        forest_rows.append({"label": level, "odds_ratio": row.odds_ratio, "low": row.low, "high": row.high})
path = charts.forest(
    forest_rows,
    "Age and body weight predict undiagnosed diabetes; so does race and ethnicity",
    "Odds of an HbA1c of 6.5% or higher among adults with no diabetes diagnosis, each factor adjusted for the others.",
    FIGURES, "odds_ratios", x_ticks=[0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
display(Image(str(path), width=720))

# %% [markdown]
# ## 5. Are the undiagnosed in contact with the health system?

# %%
undiagnosed = adults & (data.undiagnosed == 1)
contact_rows = []
for label, column in [("Has health insurance", "insured"), ("Has a usual place of care", "usual_place_of_care"),
                      ("Has been told they have high blood pressure", "told_high_blood_pressure"),
                      ("Has been told they have prediabetes", "told_prediabetes")]:
    contact_rows.append({"among adults with undiagnosed diabetes": label, **percent(design.proportion(data[column], undiagnosed))})
contact = pd.DataFrame(contact_rows)
contact.to_csv(RESULTS / "contact_with_care.csv", index=False)
display(contact)

# The same question one step earlier: of adults in the prediabetes range, how many have been told?
in_prediabetes_range = adults & (data.status == "Prediabetes range")
told = percent(design.proportion(data.told_prediabetes, in_prediabetes_range))
pd.DataFrame([{"among adults in the prediabetes range": "Has been told they have prediabetes", **told}]).to_csv(
    RESULTS / "prediabetes_awareness.csv", index=False)
print(f"Of adults in the prediabetes range, {told['estimate']:.1f}% ({told['low']:.1f} to {told['high']:.1f}) "
      f"have been told they have prediabetes (n = {told['n']:,}).")

# %% [markdown]
# ## 6. Which screening rule finds them?
#
# Each rule is applied to adults with no diagnosis. Three questions per rule: how many adults would be
# tested, what share of the undiagnosed cases are among them, and how many tests it takes to find one case.

# %%
asian = data.race_ethnicity == "Non-Hispanic Asian"
has_bmi = data.bmi.notna()
RULES = {
    "Age 35 to 70 with BMI of 25 or over (USPSTF 2021)": (data.age.between(35, 70) & (data.bmi >= 25)).where(has_bmi),
    "The same, with BMI of 23 or over for Asian adults": (data.age.between(35, 70) & ((data.bmi >= 25) | (asian & (data.bmi >= 23)))).where(has_bmi),
    "Everyone aged 35 or over": (data.age >= 35),
    "Everyone aged 45 or over": (data.age >= 45),
}
rule_rows = []
for name, rule in RULES.items():
    eligible = rule.astype(float)
    tested = percent(design.proportion(eligible, no_diagnosis))
    found = percent(design.proportion(eligible, no_diagnosis & (data.undiagnosed == 1)))
    yield_ = percent(design.proportion(data.undiagnosed, no_diagnosis & (eligible == 1)))
    rule_rows.append({"rule": name, "adults_tested_pct": tested["estimate"],
                      "cases_covered_pct": found["estimate"], "cases_covered_low": found["low"], "cases_covered_high": found["high"],
                      "cases_per_100_tests": yield_["estimate"], "tests_per_case": 100 / yield_["estimate"]})
rules = pd.DataFrame(rule_rows)
rules.to_csv(RESULTS / "screening_rules.csv", index=False)
display(rules)

# %% [markdown]
# Who does the first rule miss? Each undiagnosed case falls in exactly one of these groups.

# %%
cases = no_diagnosis & (data.undiagnosed == 1) & has_bmi
missed_rows = []
for label, group in [
    ("Covered: aged 35 to 70 with BMI of 25 or over", data.age.between(35, 70) & (data.bmi >= 25)),
    ("Missed: older than 70", data.age > 70),
    ("Missed: aged 35 to 70 with BMI under 25", data.age.between(35, 70) & (data.bmi < 25)),
    ("Missed: younger than 35", data.age < 35),
]:
    missed_rows.append({"undiagnosed cases": label, **percent(design.proportion(group.astype(float), cases))})
missed = pd.DataFrame(missed_rows)
missed.to_csv(RESULTS / "who_the_rule_misses.csv", index=False)
display(missed)

# %% [markdown]
# ## Limits
#
# - HbA1c alone is used. Surveys that also use fasting glucose find somewhat more undiagnosed diabetes.
# - One blood test is not a clinical diagnosis, which needs a repeat test.
# - "Diagnosed" is self-reported.
# - This cycle has 15 degrees of freedom, so intervals are wide and small groups cannot be reported.
#   The standard errors are themselves estimates from 30 sampling units, so an interval can come out
#   too narrow or too wide by chance; the interval for women in section 3 is an example of too narrow.
# - The data describe the United States in 2021 to 2023.
