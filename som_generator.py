import pandas as pd
import numpy as np
import joblib
import pickle

rf = joblib.load("models/rf_model.pkl")
stats = pickle.load(open("models/stats.pkl", "rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))


def compute_confidence(rf, scenario):
    tree_preds = [tree.predict(scenario.values)[0] for tree in rf.estimators_]

    mean_pred = np.mean(tree_preds)
    std_pred = np.std(tree_preds)
    return 1 - (std_pred / mean_pred)

def compute_robustness(rf, scenario, delta=0.05):
    base_pred = rf.predict(scenario)[0]
    scenario_up = scenario.copy()
    scenario_up["Quantity"] *= (1 + delta)
    pred_up = rf.predict(scenario_up)[0]
    sensitivity = abs(pred_up - base_pred) / base_pred
    return 1 - sensitivity

def compute_distribution_score(scenario, stats):
    q = scenario["Quantity"].values[0]
    p = scenario["Price"].values[0]
    z_q = abs((q - stats["Quantity"]["mean"]) / stats["Quantity"]["std"])
    z_p = abs((p - stats["Price"]["mean"]) / stats["Price"]["std"])
    ood = (z_q + z_p) / 2
    return max(np.exp(-0.5 * ood), 0.01)

def compute_reliability(conf, rob, dist):
    return (conf * rob * dist) ** (1/3)


quantities = np.linspace(1, 200, 40)
prices = np.linspace(1, 50, 40)

records = []

print("Generating reliability landscape...")

for q in quantities:
    for p in prices:

        scenario = pd.DataFrame([[0]*len(feature_columns)],
                                columns=feature_columns)

        scenario["Quantity"] = q
        scenario["Price"] = p
        scenario["Country_United Kingdom"] = 1

        conf = compute_confidence(rf, scenario)
        rob = compute_robustness(rf, scenario)
        dist = compute_distribution_score(scenario, stats)
        rel = compute_reliability(conf, rob, dist)

        records.append([conf, rob, dist, rel])

landscape = pd.DataFrame(records,
                         columns=["confidence",
                                  "robustness",
                                  "distribution",
                                  "reliability"])

landscape.to_csv("reliability_landscape.csv", index=False)

print("Reliability landscape generated successfully.")
