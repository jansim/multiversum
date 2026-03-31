from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from multiversum import Universe

settings = {
    "dimensions": {
        "death_outliers": "no_exclusion",
        "damage_outliers": "no_exclusion",
        "femininity": "female_dummy",
        "model": "negative_binomial",
        "damage_transform": "linear",
        "predictor_spec": "x_damages",
        "covariates": "none",
    }
}

universe = Universe(settings=settings)
dims = universe.dimensions


# Load original hurricane data (from Simonsohn et al. 2020)
data_path = Path(".").resolve() / "hurricanes_2015_01_09.dta"
df = pd.read_stata(data_path)

# Clean data (following the Stata do-file)
df = df.drop(columns=["masfem"])                     # drop original masfem
df = df.rename(columns={
    "masfem_mturk": "masfem",                        # use MTurk femininity rating
    "ndam15": "dam",                                  # use updated damage numbers
    "gender_mf": "female",                            # rename gender
    "alldeaths": "d",                                 # rename fatalities
})
df = df.drop(columns=["ndam"])                        # drop old damages

# Create derived variables
df["lnd"] = np.log(df["d"] + 1)                      # log(deaths + 1)
df["lndam"] = np.log(df["dam"])                       # log(damages)
df["post79"] = (df["year"] > 1979).astype(int)        # post-1979 dummy

# Standardize hurricane characteristics
df["zwin"] = (df["wind"] - df["wind"].mean()) / df["wind"].std()
df["zcat"] = (df["category"] - df["category"].mean()) / df["category"].std()
df["zmin"] = -((df["min"] - df["min"].mean()) / df["min"].std())  # reversed
df["z3"] = (df["zwin"] + df["zcat"] + df["zmin"]) / 3

# (k1) Apply death outlier exclusions
if dims["death_outliers"] == "no_exclusion":
    pass
elif dims["death_outliers"] == "drop_katrina":
    df = df[df["d"] < 1833]
elif dims["death_outliers"] == "drop_katrina_audrey":
    df = df[df["d"] < 416]
else:
    raise ValueError(f"Unknown death_outliers: {dims['death_outliers']}")

# (k2) Apply damage outlier exclusions
if dims["damage_outliers"] == "no_exclusion":
    pass
elif dims["damage_outliers"] == "drop_1_highest_damage":
    df = df[df["dam"] < 75260]
elif dims["damage_outliers"] == "drop_2_highest_damage":
    df = df[df["dam"] < 62030]
elif dims["damage_outliers"] == "drop_3_highest_damage":
    df = df[df["dam"] < 52270]
else:
    raise ValueError(f"Unknown damage_outliers: {dims['damage_outliers']}")

# (k3) Femininity operationalisation
if dims["femininity"] == "female_dummy":
    df["fem"] = df["female"]
elif dims["femininity"] == "masfem_rating":
    df["fem"] = df["masfem"]
else:
    raise ValueError(f"Unknown femininity: {dims['femininity']}")

# (k5) Damage functional form
if dims["damage_transform"] == "linear":
    df["damage"] = df["dam"].astype(float)
elif dims["damage_transform"] == "log":
    df["damage"] = df["lndam"]
else:
    raise ValueError(f"Unknown damage_transform: {dims['damage_transform']}")

# Build formula string

# (k6) Predictor specification
predictor_formulas = {
    "x_damages":          "fem * damage",
    "x_damages_pressure": "fem * damage + fem * zmin",
    "x_damages_wind":     "fem * damage + fem * zwin",
    "x_damages_category": "fem * damage + fem * zcat",
    "x_damages_all":      "fem * damage + fem * z3",
    "main_effect_only":   "fem + damage + z3",
}
if dims["predictor_spec"] not in predictor_formulas:
    raise ValueError(f"Unknown predictor_spec: {dims['predictor_spec']}")
terms = predictor_formulas[dims["predictor_spec"]]

# (k7) Year covariates
if dims["covariates"] == "none":
    pass
elif dims["covariates"] == "year_x_damages":
    terms += " + year + year:damage"
elif dims["covariates"] == "post1979_x_damages":
    terms += " + post79 + post79:damage"
else:
    raise ValueError(f"Unknown covariates: {dims['covariates']}")

# (k4) Response and family
if dims["model"] == "ols_log":
    formula = f"lnd ~ {terms}"
    family = sm.families.Gaussian()
elif dims["model"] == "negative_binomial":
    formula = f"d ~ {terms}"
    family = sm.families.NegativeBinomial(alpha=1.0)
else:
    raise ValueError(f"Unknown model: {dims['model']}")

# Fit model (robust SEs, matching Stata's ", robust")
fit = smf.glm(formula=formula, data=df, family=family).fit(cov_type="HC1")

# Extract coefficient and p-value of the first predictor term
first_pred = fit.params.index[1]  # first predictor after Intercept
beta = fit.params[first_pred]
pvalue = fit.pvalues[first_pred]

# Compute predicted values at mean covariate values for male vs female names
if dims["femininity"] == "female_dummy":
    fem_male, fem_female = 0.0, 1.0
elif dims["femininity"] == "masfem_rating":
    fem_male, fem_female = 2.53, 8.29  # mean MTurk rating for male/female names
else:
    raise ValueError(f"Unknown femininity: {dims['femininity']}")

# Build prediction DataFrames at covariate means with fem set to male/female value
pred_base = df.mean(numeric_only=True).to_frame().T

pred_male_data = pred_base.copy()
pred_male_data["fem"] = fem_male

pred_female_data = pred_base.copy()
pred_female_data["fem"] = fem_female

pred_male = fit.predict(pred_male_data).iloc[0]
pred_female = fit.predict(pred_female_data).iloc[0]

# Compute expected death difference (female - male)
if dims["model"] == "ols_log":
    # Back-transform from log scale with Wooldridge correction
    # (following Stata do-file: regress d on exp(fitted)-1, get correction factor)
    fitted_log = fit.predict(df)
    exp_fitted = np.exp(fitted_log) - 1
    ln_adjust = sm.OLS(df["d"].astype(float), exp_fitted).fit().params.iloc[0]
    edif = (np.exp(pred_female) - np.exp(pred_male)) * ln_adjust
elif dims["model"] == "negative_binomial":
    # GLM with log link: predict() already returns response-scale (expected counts)
    edif = pred_female - pred_male
else:
    raise ValueError(f"Unknown model: {dims['model']}")

# Save results
sig = 1 if pvalue < 0.05 else 0

result = pd.DataFrame({
    "beta": [beta],
    "pvalue": [pvalue],
    "sig": [sig],
    "expected_deaths_male": [pred_male],
    "expected_deaths_female": [pred_female],
    "expected_deaths_diff": [edif],
    "n": [len(df)],
})

universe.save_data(result)
