import pandas as pd
df = pd.read_csv("training_table.csv")
X = df.drop(columns=["GEOID", "suicide_rate_per_100k"])
corr = X.corr()
print(corr)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.show()

corr_pairs = (
    corr.abs()
    .unstack()
    .sort_values(ascending=False)
)

# remove self-correlations
corr_pairs = corr_pairs[corr_pairs < 1]

print(corr_pairs.head(20))