import numpy as np
import pandas as pd
import joblib
import pickle
from tqdm import tqdm

rf = joblib.load("models/rf_model.pkl")
stats = pickle.load(open("models/stats.pkl", "rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))

product_code = "21931"

quantity_range = np.linspace(1, 500, 40)  
price_range = np.linspace(1, 10, 40)      

reliability_vectors = []

def compute_confidence(rf, scenario):
    tree_preds = [tree.predict(scenario.values)[0] for tree in rf.estimators_]
    mean_pred = np.mean(tree_preds)
    std_pred = np.std(tree_preds)
    return 1 - (std_pred / mean_pred)

def compute_robustness(rf, scenario):
    base_pred = rf.predict(scenario)[0]

    scenario_up = scenario.copy()
    scenario_up["Quantity"] *= 1.05

    scenario_down = scenario.copy()
    scenario_down["Quantity"] *= 0.95

    pred_up = rf.predict(scenario_up)[0]
    pred_down = rf.predict(scenario_down)[0]

    sens_up = abs(pred_up - base_pred) / base_pred
    sens_down = abs(pred_down - base_pred) / base_pred

    S = max(sens_up, sens_down)
    return 1 - S

def compute_distribution(scenario, stats):
    q = scenario["Quantity"].values[0]
    p = scenario["Price"].values[0]

    z_q = abs((q - stats["Quantity"]["mean"]) / stats["Quantity"]["std"])
    z_p = abs((p - stats["Price"]["mean"]) / stats["Price"]["std"])

    ood = (z_q + z_p) / 2
    return np.exp(-0.5 * ood)

for q in tqdm(quantity_range):
    for p in price_range:

        scenario = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)
        scenario["Quantity"] = q
        scenario["Price"] = p
        scenario[f"StockCode_{product_code}"] = 1

        confidence = compute_confidence(rf, scenario)
        robustness = compute_robustness(rf, scenario)
        distribution = compute_distribution(scenario, stats)

        reliability = (confidence * robustness * distribution) ** (1/3)

        reliability_vectors.append([confidence, robustness, distribution, reliability])

reliability_vectors = np.array(reliability_vectors)

np.save("reliability_vectors.npy", reliability_vectors)

print("Reliability landscape generated.")