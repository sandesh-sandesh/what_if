import pandas as pd
import numpy as np
import joblib
import pickle
import sys


rf = joblib.load("models/rf_model.pkl")
stats = pickle.load(open("models/stats.pkl", "rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))

mean_vector = pickle.load(open("models/mean_vector.pkl", "rb"))
cov_matrix = pickle.load(open("models/cov_matrix.pkl", "rb"))
inv_cov_matrix = np.linalg.inv(cov_matrix)

print("\nModel loaded.")
print("Number of trees:", len(rf.estimators_))


df = pd.read_csv("cleaned_retail.csv", low_memory=False)
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

top_products = df["StockCode"].value_counts().nlargest(20).index
df = df[df["StockCode"].isin(top_products)]


product_columns = [col for col in feature_columns if col.startswith("StockCode_")]
products = [col.replace("StockCode_", "") for col in product_columns]

print("\nAvailable Products:")
for i, p in enumerate(products):
    print(f"{i+1}. {p}")

try:
    product_index = int(input("\nSelect product number: ")) - 1
except:
    print("Invalid input.")
    sys.exit()

if product_index < 0 or product_index >= len(products):
    print("Invalid product selection.")
    sys.exit()

selected_product = products[product_index]

product_data = df[df["StockCode"] == selected_product]

q_min = product_data["Quantity"].min()
q_max = product_data["Quantity"].max()
p_min = product_data["Price"].min()
p_max = product_data["Price"].max()

print("\n-----------------------------------")
print("HISTORICAL RANGE FOR THIS PRODUCT")
print("-----------------------------------")
print(f"Quantity Range : {q_min} to {q_max}")
print(f"Price Range    : £{round(p_min,2)} to £{round(p_max,2)}")
print("-----------------------------------")

quantity_input = float(input("Enter Quantity: "))
price_input = float(input("Enter Price (£): "))

if quantity_input < q_min or quantity_input > q_max:
    print("⚠ Quantity outside historical range.")

if price_input < p_min or price_input > p_max:
    print("⚠ Price outside historical range.")

scenario = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)
scenario["Quantity"] = quantity_input
scenario["Price"] = price_input
scenario[f"StockCode_{selected_product}"] = 1

if "Country_United Kingdom" in feature_columns:
    scenario["Country_United Kingdom"] = 1


tree_preds = np.array([tree.predict(scenario.values)[0] for tree in rf.estimators_])

mean_pred = np.mean(tree_preds)
std_pred = np.std(tree_preds)

lower_bound = np.percentile(tree_preds, 5)
upper_bound = np.percentile(tree_preds, 95)


confidence = 1 - (std_pred / mean_pred)


base_pred = rf.predict(scenario)[0]

scenario_up = scenario.copy()
scenario_up["Quantity"] *= 1.05
pred_up = rf.predict(scenario_up)[0]

scenario_down = scenario.copy()
scenario_down["Quantity"] *= 0.95
pred_down = rf.predict(scenario_down)[0]

sensitivity_up = abs(pred_up - base_pred) / base_pred
sensitivity_down = abs(pred_down - base_pred) / base_pred

worst_sensitivity = max(sensitivity_up, sensitivity_down)
robustness = 1 - worst_sensitivity

z_q = abs((quantity_input - stats["Quantity"]["mean"]) / stats["Quantity"]["std"])
z_p = abs((price_input - stats["Price"]["mean"]) / stats["Price"]["std"])
ood = (z_q + z_p) / 2
distribution_score = np.exp(-0.5 * ood)

x = np.array([quantity_input, price_input])
diff = x - mean_vector
distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)

geometry_score = np.exp(-0.2 * distance)


reliability = (
    confidence *
    robustness *
    distribution_score *
    geometry_score
) ** (1/4)

def interpret(value):
    if value >= 0.8:
        return "High"
    elif value >= 0.5:
        return "Moderate"
    else:
        return "Low"


print("\n==============================")
print("      SCENARIO RESULT")
print("==============================")

print(f"Product Selected : {selected_product}")
print(f"Quantity Entered : {quantity_input}")
print(f"Price Entered    : £{price_input}")

print(f"\nPredicted Revenue: £{round(mean_pred,2)}")
print(f"Prediction Interval (90%): £{round(lower_bound,2)} – £{round(upper_bound,2)}")

print("\n--- Reliability Breakdown ---")
print(f"Confidence   : {round(confidence,4)} / 1.00  ({interpret(confidence)})")
print(f"Robustness   : {round(robustness,4)} / 1.00  ({interpret(robustness)})")
print(f"Distribution : {round(distribution_score,4)} / 1.00  ({interpret(distribution_score)})")
print(f"Geometry     : {round(geometry_score,4)} / 1.00  ({interpret(geometry_score)})")

print(f"\nFinal Reliability: {round(reliability,4)} / 1.00  ({interpret(reliability)})")


print("\n--- Explanation ---")

if reliability >= 0.8:
    print("This scenario is highly reliable. The model is stable and inputs align with historical patterns.")
elif reliability >= 0.5:
    print("This scenario has moderate reliability. Some elements deviate from historical behavior.")
else:
    print("This scenario is low reliability. The inputs differ significantly from historical patterns.")

if distribution_score < 0.4:
    print("• The quantity or price is far from historical averages.")

if geometry_score < 0.4:
    print("• The quantity-price combination is historically uncommon.")

if robustness < 0.5:
    print("• Small quantity changes strongly affect prediction stability.")

if confidence < 0.5:
    print("• Model trees disagree significantly on this prediction.")

print("==============================")


print("\n==============================")
print("        ABLATION STUDY")
print("==============================")

rel_no_G = (confidence * robustness * distribution_score) ** (1/3)
rel_no_D = (confidence * robustness * geometry_score) ** (1/3)
rel_no_R = (confidence * distribution_score * geometry_score) ** (1/3)
rel_no_C = (robustness * distribution_score * geometry_score) ** (1/3)

print(f"Full Reliability         : {round(reliability,4)}")
print(f"Without Geometry         : {round(rel_no_G,4)}")
print(f"Without Distribution     : {round(rel_no_D,4)}")
print(f"Without Robustness       : {round(rel_no_R,4)}")
print(f"Without Confidence       : {round(rel_no_C,4)}")

differences = {
    "Geometry": abs(rel_no_G - reliability),
    "Distribution": abs(rel_no_D - reliability),
    "Robustness": abs(rel_no_R - reliability),
    "Confidence": abs(rel_no_C - reliability)
}

most_influential = max(differences, key=differences.get)

print("\n--- Component Influence Analysis ---")

for comp, diff in differences.items():
    print(f"{comp} impact difference: {round(diff,4)}")

print(f"\nMost Influential Component: {most_influential}")

print(f"\nMost Influential Component: {most_influential}")

print("\n--- Detailed Impact Explanation ---")

if most_influential == "Distribution":
    print("The primary driver of low reliability is deviation from historical averages.")
    print("The entered quantity or price is significantly different from what has been observed in past transactions.")
    print("This means the model is extrapolating beyond typical business conditions.")

elif most_influential == "Geometry":
    print("The primary driver of low reliability is an uncommon quantity-price combination.")
    print("Although individual values may appear reasonable, this specific combination rarely occurred historically.")
    print("This increases uncertainty in real-world outcomes.")

elif most_influential == "Robustness":
    print("The primary driver of low reliability is local sensitivity instability.")
    print("Small changes in quantity cause noticeable variation in predicted revenue.")
    print("This suggests the decision outcome may fluctuate under small operational changes.")

elif most_influential == "Confidence":
    print("The primary driver of low reliability is internal model disagreement.")
    print("The individual decision trees in the ensemble do not strongly agree on the predicted revenue.")
    print("This indicates structural uncertainty in the prediction.")

print("\nBusiness Interpretation:")
if reliability < 0.5:
    print("This scenario should be treated cautiously. It differs significantly from historical patterns.")
elif reliability < 0.8:
    print("This scenario is moderately reliable but involves some deviation from historical behavior.")
else:
    print("This scenario aligns well with historical behavior and shows strong internal stability.")


print("==============================\n")
