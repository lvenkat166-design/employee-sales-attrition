import pandas as pd

# Load inventory data
df = pd.read_csv("inventory_data.csv")

# Load sales forecast
forecast = pd.read_csv("sales_forecast.csv")

# Calculate predicted daily demand
daily_demand = forecast["yhat"].tail(7).mean()

# Calculate reorder point
df["Daily_Demand"] = daily_demand

df["Reorder_Level"] = (
    df["Daily_Demand"] *
    df["Lead_Time_Days"] +
    df["Safety_Stock"]
)

# Calculate recommended quantity
df["Order_Quantity"] = (
    df["Reorder_Level"] -
    df["Current_Stock"]
).clip(lower=0)

# Identify stock status
df["Status"] = df.apply(
    lambda row:
    "LOW STOCK"
    if row["Current_Stock"] <= row["Reorder_Level"]
    else "SUFFICIENT STOCK",
    axis=1
)

# Save report
df.to_csv(
    "inventory_report.csv",
    index=False
)

print("Inventory Analysis")
print("------------------")

print(df[[
    "Product_ID",
    "Current_Stock",
    "Reorder_Level",
    "Order_Quantity",
    "Status"
]])

print("Inventory report saved!")