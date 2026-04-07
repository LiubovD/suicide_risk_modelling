import matplotlib.pyplot as plt
import pandas as pd

# ===== INPUT YOUR RESULTS =====

poisson_vars = [
    "housing_occupied_share",
    "poverty_rate_child_u18",
    "avg_hh_size",
    "lfp_rate_16p"
]

shap_vars = [
    "pct_hispanic",   # replacing foreign born
    "veteran_share_18p",
    "marital_married_share_15p",
    "avg_hh_size",
    "rent_burdened_gt50",
    "poverty_rate_child_u18",
    "lfp_rate_16p",
    "hh_female_householder_no_spouse_share",
    "housing_occupied_share",
    "pct_age_15_24"
]

all_vars = sorted(set(poisson_vars + shap_vars))

df = pd.DataFrame({
    "Variable": all_vars,
    "Poisson": [1 if v in poisson_vars else 0 for v in all_vars],
    "SHAP": [1 if v in shap_vars else 0 for v in all_vars]
})

df.set_index("Variable").plot(kind="barh", figsize=(8,6))
plt.title("Predictor Importance: Poisson vs Machine Learning")
plt.xlabel("Selected / Important")
plt.tight_layout()
plt.show()

agreement = list(set(poisson_vars) & set(shap_vars))

plt.figure(figsize=(6,3))
plt.scatter([1]*len(poisson_vars), poisson_vars, label="Poisson", s=100)
plt.scatter([2]*len(shap_vars), shap_vars, label="SHAP", s=100)

for v in agreement:
    plt.scatter(1, v, color="red", s=150)
    plt.scatter(2, v, color="red", s=150)

plt.xticks([1,2], ["Poisson","Machine Learning"])
plt.title("Model Agreement")
plt.legend()
plt.show()

agreement = list(set(poisson_vars) & set(shap_vars))

plt.figure(figsize=(6,3))
plt.scatter([1]*len(poisson_vars), poisson_vars, label="Poisson", s=100)
plt.scatter([2]*len(shap_vars), shap_vars, label="SHAP", s=100)

for v in agreement:
    plt.scatter(1, v, color="red", s=150)
    plt.scatter(2, v, color="red", s=150)

plt.xticks([1,2], ["Poisson","Machine Learning"])
plt.title("Model Agreement")
plt.legend()
plt.show()