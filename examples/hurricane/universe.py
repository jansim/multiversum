"""
Reimplementation of the hurricane multiverse analysis from
Jung et al. / Simonsohn et al. using Python and multiversum.

Original R implementation: example-hurricane.Rmd (using the R multiverse package)

Each universe fits a GLM (Gaussian or Poisson) predicting hurricane deaths,
varying outlier exclusions, femininity operationalisation, damage transformation,
model family, interaction terms, additional predictors, and covariates.

The output is the mean difference in expected deaths between hurricanes with
female vs. male names (female=1 minus female=0).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from multiversum import Universe

# ---------------------------------------------------------------------------
# Initialise universe
# ---------------------------------------------------------------------------
universe = Universe(
    settings={
        "dimensions": {
            "death_outliers": "no_exclusion",
            "damage_outliers": "no_exclusion",
            "femininity_calculation": "masfem",
            "damage_transform": "no_transform",
            "model": "linear",
            "main_interaction": "no",
            "other_predictors": "none",
            "covariates": "1",
        }
    },
    set_seed=False,
)

dims = universe.dimensions

# ---------------------------------------------------------------------------
# Load and prepare data (mirrors the R data preparation exactly)
# ---------------------------------------------------------------------------
data_path = Path(__file__).parent / "hurricane.csv"
hurricane_data = pd.read_csv(data_path)

# Rename variables to match the R analysis
hurricane_data = hurricane_data.rename(
    columns={
        "Year": "year",
        "Name": "name",
        "NDAM": "dam",
        "alldeaths": "death",
        "Gender_MF": "female",
        "MasFem": "masfem",
        "Category": "category",
        "Minpressure_Updated_2014": "pressure",
        "HighestWindSpeed": "wind",
    }
)

# Create derived variables (pandas .std() uses ddof=1 by default, matching R's sd())
hurricane_data["post"] = (hurricane_data["year"] > 1979).astype(int)
hurricane_data["zcat"] = (
    (hurricane_data["category"] - hurricane_data["category"].mean())
    / hurricane_data["category"].std()
)
hurricane_data["zpressure"] = -(
    (hurricane_data["pressure"] - hurricane_data["pressure"].mean())
    / hurricane_data["pressure"].std()
)
hurricane_data["zwind"] = (
    (hurricane_data["wind"] - hurricane_data["wind"].mean())
    / hurricane_data["wind"].std()
)
hurricane_data["z3"] = (
    hurricane_data["zpressure"] + hurricane_data["zcat"] + hurricane_data["zwind"]
) / 3

# ---------------------------------------------------------------------------
# Apply outlier exclusions
# ---------------------------------------------------------------------------
df = hurricane_data.copy()

# Death outliers
if dims["death_outliers"] == "most_extreme_deaths":
    df = df[df["name"] != "Katrina"]
elif dims["death_outliers"] == "most_extreme_two_deaths":
    df = df[~df["name"].isin(["Katrina", "Audrey"])]

# Damage outliers
if dims["damage_outliers"] == "most_extreme_one_damage":
    df = df[~df["name"].isin(["Sandy"])]
elif dims["damage_outliers"] == "most_extreme_two_damage":
    df = df[~df["name"].isin(["Sandy", "Andrew"])]
elif dims["damage_outliers"] == "most_extreme_three_damage":
    df = df[~df["name"].isin(["Sandy", "Andrew", "Donna"])]

# ---------------------------------------------------------------------------
# Compute analysis variables
# ---------------------------------------------------------------------------
# Femininity operationalisation
if dims["femininity_calculation"] == "masfem":
    df["femininity"] = df["masfem"]
else:  # "female"
    df["femininity"] = df["female"]

# Damage transformation
if dims["damage_transform"] == "log_transform":
    df["damage"] = np.log(df["dam"])
else:  # "no_transform"
    df["damage"] = df["dam"].astype(float)

# ---------------------------------------------------------------------------
# Build GLM formula
# ---------------------------------------------------------------------------
# Response variable
if dims["model"] == "linear":
    df["response"] = np.log(df["death"] + 1)
    family = sm.families.Gaussian()
else:  # "poisson"
    df["response"] = df["death"].astype(float)
    family = sm.families.Poisson()

# Main terms
if dims["main_interaction"] == "no":
    main_terms = "femininity + damage"
else:  # "yes"
    main_terms = "femininity * damage"

# Other predictors (constraints in multiverse.toml ensure valid combinations)
other_terms = ""
if dims["other_predictors"] == "pressure":
    other_terms = " + femininity * zpressure"
elif dims["other_predictors"] == "wind":
    other_terms = " + femininity * zwind"
elif dims["other_predictors"] == "category":
    other_terms = " + femininity * zcat"
elif dims["other_predictors"] == "all":
    other_terms = " + femininity * z3"
elif dims["other_predictors"] == "all_no_interaction":
    other_terms = " + z3"

# Covariates
cov_terms = ""
if dims["covariates"] == "2":
    cov_terms = " + year:damage"
elif dims["covariates"] == "3":
    cov_terms = " + post:damage"

formula = f"response ~ {main_terms}{other_terms}{cov_terms}"

# ---------------------------------------------------------------------------
# Fit model
# ---------------------------------------------------------------------------
fit = smf.glm(formula=formula, data=df, family=family).fit()

# ---------------------------------------------------------------------------
# Compute predictions and expected deaths
# ---------------------------------------------------------------------------
pred = fit.predict(df)

if dims["model"] == "linear":
    # Back-transform from log scale with bias correction: E[Y] = exp(mu + sigma^2/2) - 1
    # sigma = residual standard deviation = sqrt(scale) for Gaussian GLM
    sigma_val = np.sqrt(fit.scale)
    expected_deaths = np.exp(pred + sigma_val**2 / 2) - 1
else:  # "poisson"
    # Predictions are already on the count scale (type="response")
    expected_deaths = pred

df = df.copy()
df["expected_deaths"] = expected_deaths

# ---------------------------------------------------------------------------
# Aggregate: mean difference in expected deaths (female=1 minus female=0)
# Mirrors R's: compare_levels(mean_deaths, by = female)
# ---------------------------------------------------------------------------
mean_by_female = df.groupby("female")["expected_deaths"].mean()
mean_deaths_diff = mean_by_female.loc[1] - mean_by_female.loc[0]

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
result = pd.DataFrame(
    {
        "mean_deaths": [mean_deaths_diff],
    }
)
universe.save_data(result)
