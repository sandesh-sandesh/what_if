import pandas as pd
import numpy as np
import joblib
import pickle
import sys

rf = joblib.load("models/rf_model.pkl")
stats = pickle.load(open("models/stats.pkl", "rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))

print("Model loaded.")
print("Number of trees:", len(rf.estimators_))

q_min = stats["Quantity"]["min"]
q_max = stats["Quantity"]["max"]
p_min = stats["Price"]["min"]
p_max = stats["Price"]["max"]

q_upper_limit = q_max * 1.5
p_upper_limit = p_max * 1.5

print("\n--- Allowed Input Range ---")
print(f"Quantity: {int(q_min)} to {int(q_upper_limit)}")
print(f"Price: {round(p_min,2)} to {round(p_upper_limit,2)}")

try:
    quantity_input = float(input("Enter Quantity: "))
    price_input = float(input("Enter Price: "))
except:
    print("Invalid input.")
    sys.exit()

if quantity_input < q_min or price_input < p_min:
    print("❌ Values below historical minimum not allowed.")
    sys.exit()

if quantity_input > q_upper_limit or price_input > p_upper_limit:
    print("❌ Value far outside safe range. Scenario rejected.")
    sys.exit()

if quantity_input > q_max:
    print("⚠ Quantity above historical max — extrapolation zone.")

if price_input > p_max:
    print("⚠ Price above historical max — extrapolation zone.")


scenario = pd.DataFrame([[0]*len(feature_columns)],
                        columns=feature_columns)

scenario["Quantity"] = quantity_input
scenario["Price"] = price_input
scenario["Country_United Kingdom"] = 1


def compute_confidence(rf, scenario):
    tree_preds = [tree.predict(scenario.values)[0] for tree in rf.estimators_]
    mean_pred = np.mean(tree_preds)
    std_pred = np.std(tree_preds)
    return mean_pred, 1 - (std_pred / mean_pred)

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


mean_pred, confidence = compute_confidence(rf, scenario)
robustness = compute_robustness(rf, scenario)
distribution_score = compute_distribution_score(scenario, stats)
reliability = compute_reliability(confidence,
                                  robustness,
                                  distribution_score)


print("\n--- SCENARIO RESULTS ---")
print("Prediction:", round(mean_pred, 2))
print("Confidence:", round(confidence, 4))
print("Robustness:", round(robustness, 4))
print("Distribution Score:", round(distribution_score, 4))
print("Final Reliability:", round(reliability, 4))
