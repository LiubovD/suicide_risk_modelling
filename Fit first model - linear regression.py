import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("training_table_removed_outliers_experiment_drop_values_by_ai.csv")

# ⭐ Use log target
y = np.log(df["suicide_rate_per_100k"] + 1)

X = df.drop(columns=[
    "GEOID",
    "suicide_rate_per_100k",
    "median_hh_income",
    "avg_hh_size",
    "housing_occupied_share"
])

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Test R²:", r2_score(y_test, pred))

# Residual plot
resid = y_test - pred
plt.scatter(pred, resid)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()