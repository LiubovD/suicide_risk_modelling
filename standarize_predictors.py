# Step 1 — Standardize predictors (LASSO-preselected)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("training_table_bn.csv")

# outcome: suicide rate per 100k
y = (df["suicide_count"] / df["pop_total"]) * 100000

# ✅ LASSO-preselected predictors (from your LASSO plot)
#selected = [
#'pct_male', 'pct_age_15_24', 'pct_age_65p', 'poverty_rate_child_u18', 'poverty_rate_elderly_65p', 'gini_index', 'unemployment_rate', 'edu_hs_grad_25p',  'hh_single_person_share', 'marital_divorced_share_15p', 'housing_occupied_share', 'rent_burdened_gt50', 'residential_stability_same_house_1yr', 'foreign_born_share', 'uninsured_rate', 'veteran_share_18p'
#]

selected = [
    "pct_male",
    "pct_age_25_44",
    "median age",
    #"pct_age_45_64",
 #   "pct_hispanic",
   "pct_nh_black",
    "median_hh_income",
    "poverty_rate_all",
 #   "poverty_rate_child_u18",
#  "poverty_rate_elderly_65p",
    "gini_index",
    "hh_public_assistance_share",
    "lfp_rate_16p",
    "unemployment_rate",
    "edu_lt_hs_25p",
    "hh_single_person_share",
    "avg_hh_size",
#    "housing_occupied_share",
  #  "renter_occupied_share",
    "overcrowding_rate_gt1pproom",
    "rent_burdened_gt50",
    "residential_stability_same_house_1yr",
    "uninsured_rate",
    "veteran_share_18p"
]

# keep only columns that actually exist (prevents KeyError if a column is missing)
selected = [c for c in selected if c in df.columns]

X = df[selected]

# Clean: remove inf, drop rows with any NA in y or X
X = X.replace([np.inf, -np.inf], np.nan)
data = pd.concat([y.rename("y"), X], axis=1).dropna()

y = data["y"]
X = data.drop(columns=["y"])

# Standardize predictors
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Step 2 — Check multicollinearity (VIF)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = sm.add_constant(X_scaled)

vif = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])]
})

print("\nVIF (sorted):")
print(vif.sort_values("VIF", ascending=False).to_string(index=False))

# Step 3 — First interpretable model (OLS on rate)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_scaled, y)

coeff = pd.Series(model.coef_, index=X.columns)
print("\nOLS coefficients on standardized predictors (sorted by |coef|):")
print(coeff.sort_values(key=np.abs, ascending=False).to_string())