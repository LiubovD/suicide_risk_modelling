import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from matplotlib.lines import Line2D

# -----------------------
# Configuration
# -----------------------
CSV_PATH = "training_table_with_density.csv"
OFFSET_COL = "pop_total"
TARGET_COL = "suicide_count"

SELECTED = [
    "pop_density",
    "housing_occupied_share",
    "hh_single_person_share",
]

PRETTY = {
    "pop_density": "Population density",
    "housing_occupied_share": "Housing occupied (%)",
    "hh_single_person_share": "Single-person households (%)",
}

ZCRIT_975 = 1.959963984540054  # N(0,1) 97.5% quantile


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# -----------------------
# Load + validate + prepare
# -----------------------
df = pd.read_csv(CSV_PATH)

missing = [c for c in SELECTED + [OFFSET_COL, TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

y = df[TARGET_COL].astype(float)
pop = df[OFFSET_COL].astype(float)

if (pop <= 0).any():
    bad = df.loc[pop <= 0, ["GEOID", OFFSET_COL]].head(10)
    raise ValueError(
        f"{OFFSET_COL} must be > 0 for log(offset). Examples:\n{bad}"
    )

X = df[SELECTED].copy()

keep = (~X.isna().any(axis=1)) & (~y.isna()) & (~pop.isna())
df_model = df.loc[keep].copy()
X = X.loc[keep].copy()
y = y.loc[keep].copy()
pop = pop.loc[keep].copy()

# Standardize predictors (IRR interpreted per 1 SD increase)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index,
)

X_model = sm.add_constant(X_scaled, has_constant="add")

# -----------------------
# Fit Poisson GLM + offset + robust SE
# -----------------------
poisson = sm.GLM(
    y,
    X_model,
    family=sm.families.Poisson(),
    offset=np.log(pop),
).fit(cov_type="HC3")

print(poisson.summary())
print("\nDeviance/df:", poisson.deviance / poisson.df_resid)

# Save deviance residuals
df_model["residuals_deviance"] = poisson.resid_deviance
df_model[["GEOID", "residuals_deviance"]].to_csv("model_residuals.csv", index=False)
print("\nResiduals saved to model_residuals.csv")


# -----------------------
# IRR computation (robust)
# -----------------------
def build_irr_long_df(model, pretty_map=None) -> pd.DataFrame:
    params = model.params.copy()
    se = model.bse.copy()  # robust SE because fit(cov_type="HC3")
    lo = params - ZCRIT_975 * se
    hi = params + ZCRIT_975 * se

    out = pd.DataFrame({
        "variable": params.index,
        "coef": params.values,
        "robust_se": se.values,
        "IRR": np.exp(params.values),
        "CI_low": np.exp(lo.values),
        "CI_high": np.exp(hi.values),
        "p_value": model.pvalues.values,
    })

    out = out[out["variable"] != "const"].copy()

    if pretty_map:
        out["label"] = out["variable"].map(lambda v: pretty_map.get(v, v))
    else:
        out["label"] = out["variable"]

    out["pct_change"] = (out["IRR"] - 1.0) * 100.0
    out = out.sort_values("p_value").reset_index(drop=True)
    return out


def save_poster_table(irr_long_df: pd.DataFrame, save_path="irr_results_poster.csv") -> pd.DataFrame:
    t = irr_long_df.copy()
    poster_table = pd.DataFrame({
        "Predictor": t["label"],
        "IRR": t["IRR"].map(lambda x: f"{x:.2f}"),
        "95% CI": t.apply(lambda r: f"{r['CI_low']:.2f}–{r['CI_high']:.2f}", axis=1),
        "p": t["p_value"].map(format_p),
        "% change": t["pct_change"].map(lambda x: f"{x:+.0f}%"),
    })
    poster_table.to_csv(save_path, index=False)
    print(f"Saved poster table: {save_path}")
    return poster_table


irr_long = build_irr_long_df(poisson, pretty_map=PRETTY)
irr_long.to_csv("irr_results_long.csv", index=False)
print("Saved: irr_results_long.csv")
poster_table = save_poster_table(irr_long, save_path="irr_results_poster.csv")


# -----------------------
# Forest plots
# -----------------------
def forest_plot_basic(
    irr_long_df: pd.DataFrame,
    title="Incidence Rate Ratios (IRR) with 95% CI",
    subtitle="Predictors standardized (1 SD). Population offset. Robust SE (HC3).",
    save_path="irr_forest.png",
    dpi=300,
):
    t = irr_long_df.sort_values("p_value", ascending=True).iloc[::-1].reset_index(drop=True)

    y = np.arange(len(t))
    irr = t["IRR"].values
    low = t["CI_low"].values
    high = t["CI_high"].values

    fig_h = max(3.8, 0.55 * len(t) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    xerr = np.vstack([irr - low, high - irr])
    ax.errorbar(irr, y, xerr=xerr, fmt="o", capsize=4, elinewidth=2, markersize=8)
    ax.axvline(1.0, linestyle="--", linewidth=2)

    ax.set_yticks(y)
    ax.set_yticklabels(t["label"].values, fontsize=13)
    ax.set_xscale("log")
    ax.set_xlabel("IRR (log scale)", fontsize=13)
    ax.set_title(title, fontsize=17, pad=12)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=11, va="bottom")

    # annotate using axes-fraction x so it doesn’t clip unpredictably
    for i, pv in enumerate(t["p_value"].values):
        ax.text(1.01, i, f"p={format_p(pv)}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10)

    ax.set_xlim(left=max(0.4, low.min() * 0.9), right=high.max() * 1.35)
    ax.grid(axis="x", linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def forest_plot_journal(
    irr_long_df: pd.DataFrame,
    title="Predictors of Suicide Counts",
    subtitle="",
    save_png="irr_forest_journal.png",
    save_pdf="irr_forest_journal.pdf",
    dpi=300,
):
    t = irr_long_df.sort_values("p_value", ascending=True).iloc[::-1].reset_index(drop=True)

    y = np.arange(len(t))
    irr = t["IRR"].values
    low = t["CI_low"].values
    high = t["CI_high"].values
    pvals = t["p_value"].values

    fig_h = max(3.4, 0.5 * len(t) + 1.8)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))

    colors = np.where(irr >= 1.0, "tab:red", "tab:blue")

    for i in range(len(t)):
        ax.errorbar(
            irr[i], y[i],
            xerr=[[irr[i] - low[i]], [high[i] - irr[i]]],
            fmt="o",
            capsize=4,
            elinewidth=2.0,
            markersize=7.5,
            color=colors[i],
            markerfacecolor=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.6,
        )

    ax.axvline(1.0, linestyle="--", linewidth=1.6, color="black", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(t["label"].values, fontsize=12)
    ax.set_xscale("log")
    ax.set_xlabel("Incidence Rate Ratio (log scale)", fontsize=12)

    ax.set_title(title, fontsize=16, pad=10)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, va="bottom")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    left = max(0.5, low.min() * 0.9)
    right = high.max() * 1.35
    ax.set_xlim(left=left, right=right)

    # p-values on right using axis-fraction x; y in data coords
    for i, pv in enumerate(pvals):
        ax.text(1.01, y[i], f"p={format_p(pv)}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=10)

    legend_elems = [
        Line2D([0], [0], marker='o', color='tab:red', label='Risk ↑ (IRR>1)',
               markerfacecolor='tab:red', markeredgecolor='black', markersize=7, linewidth=0),
        Line2D([0], [0], marker='o', color='tab:blue', label='Protective ↓ (IRR<1)',
               markerfacecolor='tab:blue', markeredgecolor='black', markersize=7, linewidth=0),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_png}")
    print(f"Saved: {save_pdf}")


forest_plot_basic(irr_long, save_path="irr_forest_poster.png", dpi=300)
forest_plot_journal(irr_long)