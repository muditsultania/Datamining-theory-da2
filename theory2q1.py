import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
dataset_path = "crime_data_indian.csv"
crime_data = pd.read_csv(dataset_path)

# Step 1: Preprocessing - Aggregate data
crime_data['Date'] = pd.to_datetime(crime_data['Date'])
daily_crime = crime_data.groupby('Date')['Reported_Incidents'].sum()

# Step 2: Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(daily_crime, label="Reported Incidents", color="blue")
plt.title("Daily Reported Incidents")
plt.xlabel("Date")
plt.ylabel("Number of Incidents")
plt.legend()
plt.show()

# Step 3: Stationarity Check
# Differencing to make the series stationary
daily_crime_diff = daily_crime.diff().dropna()

# Plot ACF and PACF for parameter identification
plot_acf(daily_crime_diff, lags=40)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plot_pacf(daily_crime_diff, lags=40)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# Step 4: Fit ARIMA Model (using initial parameters based on ACF/PACF analysis)
model = ARIMA(daily_crime, order=(1, 1, 1))  # (p, d, q)
arima_result = model.fit()

# Step 5: Summarize ARIMA Results
print(arima_result.summary())

# Step 6: Forecasting
forecast_steps = 30  # Forecast for the next 30 days
forecast = arima_result.forecast(steps=forecast_steps)
forecast_index = pd.date_range(daily_crime.index[-1], periods=forecast_steps + 1, freq="D")[1:]

# Visualize the forecast
plt.figure(figsize=(12, 6))
plt.plot(daily_crime, label="Historical Data", color="blue")
plt.plot(forecast_index, forecast, label="Forecast", color="orange", linestyle="--")
plt.title("ARIMA Forecast of Reported Incidents")
plt.xlabel("Date")
plt.ylabel("Number of Incidents")
plt.legend()
plt.show()
