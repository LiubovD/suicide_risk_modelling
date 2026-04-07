import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

df = pd.read_csv("training_table_bn.csv")

predictors = [

    "housing_occupied_share",
    "poverty_rate_child_u18",
    "avg_hh_size",
    "hh_female_householder_no_spouse_share",
    "foreign_born_share",
    "rent_burdened_gt50",
    "pct_age_15_24",
    "lfp_rate_16p",
    "veteran_share_18p",
    "marital_married_share_15p"
]
# You can trim this list further if you want a smaller final model.

y = df["suicide_count"].astype(int)
pop = df["pop_total"].astype(float)

X = df[predictors].copy()


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=df.index)


X_nb = sm.add_constant(X_scaled)
offset = np.log(pop)

nb = sm.NegativeBinomial(
    y,
    X_nb,
    offset=offset
).fit()

print(nb.summary())

params = nb.params
conf = nb.conf_int()
irr = np.exp(params)
irr_ci = np.exp(conf)

results = pd.DataFrame({
    "coef": params,
    "IRR": irr,
    "IRR_low": irr_ci[0],
    "IRR_high": irr_ci[1],
    "pvalue": nb.pvalues
}).sort_values("IRR", ascending=False)

print(results)
print("NB AIC:", nb.aic)