import pandas as pd
import numpy as np

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

# Aggregate
season_summary = df.groupby(["StockCode", "Season"])["Quantity"].sum().reset_index()

# Calculate seasonal index
season_index = []

for product in season_summary["StockCode"].unique():
    product_data = season_summary[season_summary["StockCode"] == product]
    avg_qty = product_data["Quantity"].mean()

    for _, row in product_data.iterrows():
        index_value = row["Quantity"] / avg_qty
        season_index.append([
            product,
            row["Season"],
            round(index_value, 2)
        ])

season_index_df = pd.DataFrame(
    season_index,
    columns=["StockCode", "Season", "Seasonal_Index"]
)

print(season_index_df)
