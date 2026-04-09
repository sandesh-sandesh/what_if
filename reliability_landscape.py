import numpy as np
import pandas as pd
import joblib
import pickle

# Load model and stats
rf = joblib.load("models/rf_model.pkl")
stats = pickle.load(open("models/stats.pkl", "rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))

# Select one product for landscape
product_columns = [col for col in feature_columns if col.startswith("StockCode_")]
selected_product = product_columns[0]  # first product

# Create grid of Quantity and Price
quantity_range = np.linspace(1, 500, 40)
price_range = np.linspace(0.5, 10, 40)

reliability_vectors = []

for q in quantity_range:
    for p in price_range:

        scenario = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)
        scenario["Quantity"] = q
        scenario["Price"] = p
        scenario[selected_product] = 1

        # Confidence
        tree_preds = [tree.predict(scenario.values)[0] for tree in rf.estimators_]
        mean_pred = np.mean(tree_preds)
        std_pred = np.std(tree_preds)
        confidence = 1 - (std_pred / mean_pred)

        # Robustness (+5 and -5)
        scenario_up = scenario.copy()
        scenario_up["Quantity"] *= 1.05
        pred_up = rf.predict(scenario_up)[0]

        scenario_down = scenario.copy()
        scenario_down["Quantity"] *= 0.95
        pred_down = rf.predict(scenario_down)[0]

        base_pred = rf.predict(scenario)[0]

        s1 = abs(pred_up - base_pred) / base_pred
        s2 = abs(pred_down - base_pred) / base_pred
        sensitivity = max(s1, s2)
        robustness = 1 - sensitivity

        # Distribution
        z_q = abs((q - stats["Quantity"]["mean"]) / stats["Quantity"]["std"])
        z_p = abs((p - stats["Price"]["mean"]) / stats["Price"]["std"])
        ood = (z_q + z_p) / 2
        distribution = np.exp(-0.5 * ood)

        # Geometry (Mahalanobis)
        v = np.array([q, p])
        mu = np.array([stats["Quantity"]["mean"], stats["Price"]["mean"]])
        cov = np.array(stats["covariance"])
        inv_cov = np.linalg.inv(cov)
        d_m = np.dot(np.dot((v - mu).T, inv_cov), (v - mu))
        geometry = np.exp(-0.2 * d_m)

        # Final reliability
        reliability = (confidence * robustness * distribution * geometry) ** 0.25

        reliability_vectors.append([confidence, robustness, distribution, geometry, reliability])

reliability_vectors = np.array(reliability_vectors)

np.save("reliability_vectors.npy", reliability_vectors)

print("Reliability landscape generated successfully.")