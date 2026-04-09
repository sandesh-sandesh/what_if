import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

print("Loading dataset...")

os.makedirs("models", exist_ok=True)

df = pd.read_csv("cleaned_retail.csv", low_memory=False)

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

print("Rows after cleaning:", len(df))

df["Revenue"] = df["Quantity"] * df["Price"]


top_products = df["StockCode"].value_counts().nlargest(20).index
df = df[df["StockCode"].isin(top_products)]

print("Rows after product filtering:", len(df))

df = pd.get_dummies(df, columns=["Country", "StockCode"])

X = df.drop(columns=[
    "Revenue",
    "Invoice",
    "Description",
    "Customer ID",
    "InvoiceDate"
], errors="ignore")

y = df["Revenue"]

print("Final feature shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Training complete.")

joblib.dump(rf, "models/rf_model.pkl")
pickle.dump(X.columns.tolist(), open("models/feature_columns.pkl", "wb"))

quantity_mean = df["Quantity"].mean()
quantity_std = df["Quantity"].std()

price_mean = df["Price"].mean()
price_std = df["Price"].std()

numeric_features = df[["Quantity", "Price"]]

mean_vector = numeric_features.mean().values
cov_matrix = np.cov(numeric_features.values, rowvar=False)

cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

stats = {
    "Quantity": {
        "mean": quantity_mean,
        "std": quantity_std
    },
    "Price": {
        "mean": price_mean,
        "std": price_std
    },
    "covariance": cov_matrix.tolist(),  
    "mean_vector": mean_vector.tolist()
}

pickle.dump(stats, open("models/stats.pkl", "wb"))

pickle.dump(mean_vector, open("models/mean_vector.pkl", "wb"))
pickle.dump(cov_matrix, open("models/cov_matrix.pkl", "wb"))

print("Model saved.")
print("Stats saved with covariance.")
print("Mahalanobis components saved.")
print("Training pipeline completed successfully.")