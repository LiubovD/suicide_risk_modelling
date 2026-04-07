import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# df = pd.read_csv("training_table_removed_outliers.csv")
# df.describe()

#
# sns.histplot(df["suicide_rate_per_100k"], bins=30)
# plt.show()
#
# sns.boxplot(x=df["suicide_rate_per_100k"])
# plt.show()
#
# df.sort_values("suicide_rate_per_100k", ascending=False).head(10)
#

# df["log_suicide_rate"] = np.log(df["suicide_rate_per_100k"] + 1)
#
# sns.histplot(df["log_suicide_rate"], bins=30)
# plt.show()
#
# sns.boxplot(x=df["log_suicide_rate"])
# plt.show()
#
#
#
#


df_count = pd.read_csv("training_table_bn.csv")


y = df_count["suicide_count"]

sns.histplot(y, bins=30)
plt.show()

sns.boxplot(y)
plt.show()

mean = df_count["suicide_count"].mean()
var = df_count["suicide_count"].var()

print("Mean:", mean)
print("Variance:", var)

import statsmodels.api as sm
import numpy as np

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# Load data
df = pd.read_csv("training_table_bn_no_outliers.csv")

# -----------------------
# Define variables
# -----------------------
# -----------------------
# Define variables
# -----------------------
selected = [
    "housing_occupied_share",
    "poverty_rate_child_u18",
    "avg_hh_size"
]

y = df["suicide_count"]
pop = df["pop_total"]
X = df[selected]

# -----------------------
# Standardize predictors
# -----------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=df.index
)

# Add intercept
X_model = sm.add_constant(X_scaled)

# -----------------------
# Fit Poisson
# -----------------------
poisson = sm.GLM(
    y,
    X_model,
    family=sm.families.Poisson(),
    offset=np.log(pop)
).fit()

# -----------------------
# Fit Negative Binomial
# -----------------------
nb = sm.GLM(
    y,
    X_model,
    family=sm.families.NegativeBinomial(),
    offset=np.log(pop)
).fit()

# -----------------------
# Likelihood Ratio Test
# -----------------------
lr_stat = 2 * (nb.llf - poisson.llf)
df_diff = nb.df_model - poisson.df_model
p_value = chi2.sf(lr_stat, df_diff)

print("\n===== MODEL COMPARISON =====")
print("Poisson logLik:", poisson.llf)
print("NB logLik:", nb.llf)
print("LR statistic:", lr_stat)
print("p-value:", p_value)

# -----------------------
# AIC comparison
# -----------------------
print("\nPoisson AIC:", poisson.aic)
print("NB AIC:", nb.aic)

# -----------------------
# Dispersion check
# -----------------------
pearson_disp = sum(poisson.resid_pearson**2) / poisson.df_resid
print("\nPearson dispersion:", pearson_disp)

influence = poisson.get_influence()
cooks = influence.cooks_distance[0]


plt.stem(cooks)
plt.title("Cook's distance")
plt.show()