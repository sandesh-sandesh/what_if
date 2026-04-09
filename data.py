import pandas as pd

df = pd.read_csv("cleaned_retail.csv")

df = pd.get_dummies(df, columns=["Country"])

y = df["Revenue"]

X = df.drop(columns=[
    "Revenue",
    "Invoice",
    "StockCode",
    "Description",
    "InvoiceDate",
    "Customer ID"
])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X columns:")
print(X.columns)
