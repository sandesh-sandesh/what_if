import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pickle

df = pd.read_csv("cleaned_retail.csv", low_memory=False)

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df["Revenue"] = df["Quantity"] * df["Price"]

top_products = df["StockCode"].value_counts().head(20).index
df = df[df["StockCode"].isin(top_products)]

df = pd.get_dummies(df, columns=["StockCode", "Country"])

X = df.drop(columns=["Revenue"])

X = X.select_dtypes(include=[np.number])

y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt"]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

print("Running Grid Search...")
grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

pickle.dump(best_model, open("rf_optimized.pkl", "wb"))

print("\nOptimized model saved as rf_optimized.pkl")