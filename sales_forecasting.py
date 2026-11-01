import pandas as pd

import matplotlib.pyplot as plt

from prophet import Prophet

# Load sales data
df = pd.read_csv("sales_data.csv")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Select product
product = "P101"

product_data = df[
    df["Product_ID"] == product
]

# Prepare Prophet columns
data = product_data[[
    "Date",
    "Quantity_Sold"
]].rename(columns={
    "Date": "ds",
    "Quantity_Sold": "y"
})

# Sort dates
data = data.sort_values("ds")

# Train Prophet
model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=False,
    daily_seasonality=False
)

model.fit(data)

# Forecast next 7 days
future = model.make_future_dataframe(
    periods=7
)

forecast = model.predict(future)

# Save forecast
forecast[[
    "ds",
    "yhat",
    "yhat_lower",
    "yhat_upper"
]].to_csv(
    "sales_forecast.csv",
    index=False
)

# Plot
model.plot(forecast)

plt.title("Retail Sales Forecast")

plt.savefig("sales_forecast.png")

plt.show()

print("Sales forecasting completed!")
print(forecast.tail(7)[["ds", "yhat"]])