"""
XGBoost regression on suicide rate per 100k + SHAP explanations
(using LASSO-selected predictors)

Outputs:
- shap_importance_bar.png
- shap_summary_beeswarm.png
- shap_dependence_<feature>.png (top K)
- prints Mean |SHAP| table
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main(
    input_csv: str = "training_table_bn_no_outliers.csv",
    test_size: float = 0.25,
    random_state: int = 42,
    top_n_bar: int = 15,
    top_k_dependence: int = 3,
):
    # -----------------------------
    # 1) Load data
    # -----------------------------
    df = pd.read_csv(input_csv)

    # -----------------------------
    # 2) Outcome: rate per 100k
    # -----------------------------
    y = ((df["suicide_count"] / df["pop_total"]) * 100000).astype(float)

    # -----------------------------
    # 3) LASSO-selected predictors
    # -----------------------------
    LASSO_FEATURES = [
        "housing_occupied_share",
        "poverty_rate_child_u18",
        "avg_hh_size",
        "hh_female_householder_no_spouse_share",
        "foreign_born_share",
        "rent_burdened_gt50",
        "pct_age_15_24",
        "lfp_rate_16p",
        "veteran_share_18p",
        "marital_married_share_15p",
    ]

    missing = [c for c in LASSO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    X = df[LASSO_FEATURES].copy()

    # -----------------------------
    # 4) Clean rows (drop NA/inf)
    # -----------------------------
    X = X.replace([np.inf, -np.inf], np.nan)
    data = pd.concat([y.rename("rate_per_100k"), X], axis=1).dropna()

    y = data["rate_per_100k"].values
    X = data.drop(columns=["rate_per_100k"])

    # -----------------------------
    # 5) Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # -----------------------------
    # 6) Scale predictors
    # -----------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # DataFrame versions (keeps column names for plotting)
    X_train_s_df = pd.DataFrame(X_train_s, columns=X.columns, index=X_train.index)
    X_test_s_df = pd.DataFrame(X_test_s, columns=X.columns, index=X_test.index)

    # -----------------------------
    # 7) Fit XGBoost model
    # -----------------------------
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        objective="reg:squarederror",
    )
    model.fit(X_train_s_df, y_train)

    # -----------------------------
    # 8) Evaluate (optional but useful)
    # -----------------------------
    pred = model.predict(X_test_s_df)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)

    print("\n--- Test metrics ---")
    print(f"R^2 :  {r2:.3f}")
    print(f"MAE :  {mae:.3f}")
    print(f"RMSE:  {rmse:.3f}")

    # -----------------------------
    # 9) SHAP explanation
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s_df)

    # -----------------------------
    # 10) Mean |SHAP| importance table + bar plot
    # -----------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    imp = pd.Series(mean_abs_shap, index=X_test_s_df.columns).sort_values(ascending=False)

    print("\nMean |SHAP| importance:")
    print(imp.to_string())

    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "mean_abs_shap"]

    top_n = min(top_n_bar, len(imp_df))
    plot_df = imp_df.head(top_n).iloc[::-1]  # reverse for barh

    plt.figure(figsize=(9, max(4, 0.45 * top_n + 1)))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"Global Feature Importance (Top {top_n})")
    plt.tight_layout()
    plt.savefig("shap_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\nSaved: shap_importance_bar.png")

    # -----------------------------
    # 11) SHAP summary beeswarm plot
    # -----------------------------
    # Use matplotlib savefig since shap.summary_plot doesn't have a save arg
    shap.summary_plot(shap_values, X_test_s_df, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved: shap_summary_beeswarm.png")

    # -----------------------------
    # 12) Optional: dependence plots for top K features
    # -----------------------------
    top_k = min(top_k_dependence, len(imp.index))
    top_features = imp.index[:top_k].tolist()

    for feat in top_features:
        shap.dependence_plot(feat, shap_values, X_test_s_df, show=False)
        plt.tight_layout()
        out = f"shap_dependence_{feat}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()
        print("Saved:", out)


if __name__ == "__main__":
    main()