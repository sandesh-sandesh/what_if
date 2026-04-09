import pandas as pd
import numpy as np
import joblib
import pickle
import sys
from datetime import datetime


rf = joblib.load("models/rf_model.pkl")
stats = pickle.load(open("models/stats.pkl", "rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))
som = pickle.load(open("models/som_model.pkl", "rb"))


df = pd.read_csv("cleaned_retail.csv", low_memory=False)

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["Month"].apply(get_season)

top_products = df["StockCode"].value_counts().nlargest(20).index
df = df[df["StockCode"].isin(top_products)]


season_summary = df.groupby(["StockCode", "Season"])["Quantity"].sum().reset_index()

season_index = {}

for product in season_summary["StockCode"].unique():
    product_data = season_summary[season_summary["StockCode"] == product]
    avg_qty = product_data["Quantity"].mean()

    season_index[product] = {}
    for _, row in product_data.iterrows():
        season_index[product][row["Season"]] = row["Quantity"] / avg_qty


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

quantity_upper_limit = q_max * 2
price_upper_limit = p_max * 1.5

print("\n-----------------------------------")
print("HISTORICAL RANGE FOR THIS PRODUCT")
print("-----------------------------------")
print(f"Quantity Range : {q_min} to {q_max}")
print(f"Price Range    : £{round(p_min,2)} to £{round(p_max,2)}")
print("-----------------------------------")


try:
    quantity_input = float(input("Enter Quantity: "))
    price_input = float(input("Enter Price (£): "))
    future_date_input = input("Enter Future Date (YYYY-MM-DD): ")
    future_date = datetime.strptime(future_date_input, "%Y-%m-%d")
except:
    print("Invalid input format.")
    sys.exit()

if price_input > price_upper_limit:
    print("❌ Price too far beyond realistic market range.")
    print("Please adjust price closer to historical levels.")
    sys.exit()

if quantity_input > quantity_upper_limit:
    print("⚠ Quantity significantly above historical demand.")

if quantity_input < q_min or quantity_input > q_max:
    print("⚠ Quantity outside historical range.")

if price_input < p_min or price_input > p_max:
    print("⚠ Price outside historical range.")


future_month = future_date.month
future_season = get_season(future_month)

season_multiplier = season_index[selected_product][future_season]

adjusted_quantity = quantity_input * season_multiplier
projected_revenue = adjusted_quantity * price_input


scenario = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)

scenario["Quantity"] = adjusted_quantity
scenario["Price"] = price_input
scenario[f"StockCode_{selected_product}"] = 1

if "Country_United Kingdom" in feature_columns:
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
reliability = compute_reliability(confidence, robustness, distribution_score)


print("\n==============================")
print("   TIME-AWARE SCENARIO RESULT")
print("==============================")

print(f"Product            : {selected_product}")
print(f"Future Date        : {future_date_input}")
print(f"Detected Season    : {future_season}")
print(f"Season Multiplier  : {round(season_multiplier,2)}")

print(f"\nBase Quantity      : {quantity_input}")
print(f"Adjusted Quantity  : {round(adjusted_quantity,2)}")
print(f"Price              : £{price_input}")

print(f"\nProjected Revenue  : £{round(projected_revenue,2)}")

if reliability > 0.9:
    risk_level = "🟢 VERY SAFE"
elif reliability > 0.75:
    risk_level = "🟢 SAFE"
elif reliability > 0.6:
    risk_level = "🟡 MODERATE RISK"
else:
    risk_level = "🔴 HIGH RISK"

print(f"\nReliability Level  : {risk_level}")
print(f"Final Reliability  : {round(reliability,2)} / 1.00")

print("\n------------------------------")
print("RELIABILITY BREAKDOWN")
print("------------------------------")

print(f"Confidence (Model Agreement)     : {round(confidence,2)}")
print(f"Robustness (Sensitivity Stability): {round(robustness,2)}")
print(f"Distribution Proximity            : {round(distribution_score,2)}")

print("==============================\n")