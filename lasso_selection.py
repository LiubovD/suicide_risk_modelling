"""
FULL LASSO SCRIPT (Poisson GLM + L1 / LASSO) on suicide rate per 100k
+ saves results table
+ 3 plots using PRETTY NAMES:
  1) selected coefficients bar chart
  2) CV error vs alpha
  3) coefficient paths (legend uses pretty names for selected vars)

Outputs:
- lasso_selected_variables.csv
- lasso_coefficients_pretty.png
- lasso_cv_error.png
- lasso_path_pretty.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_poisson_deviance

# -------------------------
# 1) Load data
# -------------------------
df = pd.read_csv("training_table_bn.csv")

y = df["suicide_count"].astype(float)
pop = df["pop_total"].astype(float)

# Guard against invalid offset
if (pop <= 0).any():
    bad = df.loc[pop <= 0, ["GEOID", "pop_total"]].head(10)
    raise ValueError(
        "pop_total must be > 0 for log(offset). Examples of bad rows:\n"
        f"{bad}"
    )

# -------------------------
# 2) Candidate predictors = ALL variables (numeric) except exclude
# -------------------------
exclude = {"GEOID", "suicide_count", "pop_total", "hh_total" }

cand_cols = [c for c in df.columns if c not in exclude]

# keep numeric only (drops objects/strings)
X = df[cand_cols].select_dtypes(include=[np.number]).copy()

# If you want to explicitly drop columns that are all-NA or constant:
X = X.dropna(axis=1, how="all")
const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if const_cols:
    X = X.drop(columns=const_cols)

# drop rows with any NA in X/y/pop (simple approach)
keep = (~X.isna().any(axis=1)) & (~y.isna()) & (~pop.isna())
X = X.loc[keep].copy()
y = y.loc[keep].copy()
pop = pop.loc[keep].copy()

print(f"Using {X.shape[1]} candidate predictors after cleaning.")

# -------------------------
# 3) Pretty names (auto)
# -------------------------
def prettify(name: str) -> str:
    return name.replace("_", " ").replace("pct ", "% ").title()

PRETTY = {c: prettify(c) for c in X.columns}
PRETTY.update({
    # optionally override a few to look nicer:
    "pct_age_25_44": "% Age 25–44",
    "pct_age_45_64": "% Age 45–64",
    "lfp_rate_16p": "Labor force participation (16+)",
    "edu_lt_hs_25p": "% < HS (25+)",
    "overcrowding_rate_gt1pproom": "Overcrowding (>1 p/room)",
    "rent_burdened_gt50": "Rent burdened (>50%)",
    "residential_stability_same_house_1yr": "Same house 1+ yr",
    "veteran_share_18p": "Veteran share (18+)",
    "median_hh_income": "Median HH income",
    "poverty_rate_all": "Poverty rate (all)",
})

# -------------------------
# 4) Train/test split
# -------------------------
X_train, X_test, y_train, y_test, pop_train, pop_test = train_test_split(
    X, y, pop, test_size=0.2, random_state=42
)

# -------------------------
# 5) Scale predictors
# -------------------------
scaler = StandardScaler()
Xtr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
Xte = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

Xtr_const = sm.add_constant(Xtr, has_constant="add")
Xte_const = sm.add_constant(Xte, has_constant="add")

# -------------------------
# 6) LASSO search over alpha using test Poisson deviance
# -------------------------
alphas = np.logspace(-4, 1, 25)
best = {"alpha": None, "deviance": np.inf, "model": None}
test_devs = []

for a in alphas:
    m = sm.GLM(
        y_train, Xtr_const,
        family=sm.families.Poisson(),
        offset=np.log(pop_train)
    ).fit_regularized(
        method="elastic_net",
        alpha=a,
        L1_wt=1.0,
        maxiter=5000
    )

    eta = np.dot(Xte_const, m.params)
    mu = np.exp(eta + np.log(pop_test))

    dev = mean_poisson_deviance(y_test, mu)
    test_devs.append(dev)

    if dev < best["deviance"]:
        best = {"alpha": a, "deviance": dev, "model": m}

print("Best alpha:", best["alpha"])
print("Best test Poisson deviance:", best["deviance"])

lasso_params = best["model"].params

selected_vars = [
    v for v in lasso_params.index
    if v != "const" and abs(lasso_params[v]) > 1e-8
]

print(f"Selected vars (LASSO): {selected_vars}")
print(f"Number selected: {len(selected_vars)}")

# -------------------------
# 7) Final unpenalized Poisson model on selected vars (scaled) + robust SE
# -------------------------
X_sel = X[selected_vars].copy()
X_sel_scaled = pd.DataFrame(
    StandardScaler().fit_transform(X_sel),
    columns=X_sel.columns,
    index=X_sel.index
)
X_model = sm.add_constant(X_sel_scaled, has_constant="add")

poisson_final = sm.GLM(
    y, X_model,
    family=sm.families.Poisson(),
    offset=np.log(pop)
).fit(cov_type="HC3")

print(poisson_final.summary())
print("Deviance/df:", poisson_final.deviance / poisson_final.df_resid)

# -------------------------
# 8) Save results table
# -------------------------
results = (
    poisson_final.params.rename("coef").to_frame()
    .assign(se=poisson_final.bse, z=poisson_final.tvalues, p=poisson_final.pvalues)
)
results["pretty"] = results.index.map(lambda c: PRETTY.get(c, c))
results.to_csv("lasso_selected_variables.csv", index=True)

# -------------------------
# 9) Plot 1: selected coefficients (pretty names)
# -------------------------
coef_no_const = results.drop(index="const", errors="ignore").copy()
coef_no_const = coef_no_const.reindex(coef_no_const["coef"].abs().sort_values(ascending=True).index)

plt.figure(figsize=(8, max(4, 0.35 * len(coef_no_const))))
plt.barh(coef_no_const["pretty"], coef_no_const["coef"])
plt.axvline(0, linewidth=1)
plt.title("Selected coefficients (final Poisson model)")
plt.tight_layout()
plt.savefig("lasso_coefficients_pretty.png", dpi=200)
plt.close()

# -------------------------
# 10) Plot 2: deviance vs alpha
# -------------------------
plt.figure(figsize=(7, 4))
plt.semilogx(alphas, test_devs, marker="o")
plt.axvline(best["alpha"], linestyle="--")
plt.title("Test Poisson deviance vs alpha")
plt.xlabel("alpha (L1 penalty)")
plt.ylabel("Mean Poisson deviance (test)")
plt.tight_layout()
plt.savefig("lasso_cv_error.png", dpi=200)
plt.close()

# -------------------------
# 11) Plot 3: coefficient paths (approx)
# -------------------------
coefs_path = []

for a in alphas:
    m = sm.GLM(
        y_train, Xtr_const,
        family=sm.families.Poisson(),
        offset=np.log(pop_train)
    ).fit_regularized(
        method="elastic_net",
        alpha=a,
        L1_wt=1.0,
        maxiter=5000
    )
    coefs_path.append(m.params.reindex(Xtr_const.columns).fillna(0.0))

coefs_path = pd.DataFrame(coefs_path, index=alphas)

plt.figure(figsize=(8, 5))
for col in coefs_path.columns:
    if col == "const":
        continue
    plt.semilogx(coefs_path.index, coefs_path[col], label=PRETTY.get(col, col), linewidth=1)

plt.axvline(best["alpha"], linestyle="--")
plt.title("Coefficient paths (Poisson GLM LASSO)")
plt.xlabel("alpha (L1 penalty)")
plt.ylabel("Coefficient")

# Only label selected vars to keep legend readable
handles, labels = plt.gca().get_legend_handles_labels()
sel_set = set(selected_vars)
filtered = [(h, l) for h, l in zip(handles, labels) if l in {PRETTY.get(k, k) for k in sel_set}]
if filtered:
    fh, fl = zip(*filtered)
    plt.legend(fh, fl, fontsize=8, loc="best")

plt.tight_layout()
plt.savefig("lasso_path_pretty.png", dpi=200)
plt.close()