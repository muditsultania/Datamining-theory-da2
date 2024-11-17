import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'crime_data_indian.csv'  # Update the path if needed
data = pd.read_csv(file_path)

# Convert Date to datetime and aggregate data by Date
data['Date'] = pd.to_datetime(data['Date'])
daily_crime = data.groupby('Date')['Reported_Incidents'].sum()

# Set window size for moving averages
window_size = 10

# 1. Moving Average (Simple Moving Average)
sma = daily_crime.rolling(window=window_size).mean()

# 2. Exponential Moving Average (EMA)
ema = daily_crime.ewm(span=window_size, adjust=False).mean()

# 3. Weighted Moving Average (WMA)
weights = np.arange(1, window_size + 1)
wma = daily_crime.rolling(window=window_size).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# 4. Hull Moving Average (HMA)
def hull_moving_average(series, window):
    wma1 = series.rolling(window=window).mean()
    wma2 = series.rolling(window=window//2).mean()
    diff = 2 * wma2 - wma1
    return diff.rolling(window=int(np.sqrt(window))).mean()

hma = hull_moving_average(daily_crime, window_size)

# 5. Moving Average Crossover
short_window = 5
long_window = 20
sma_short = daily_crime.rolling(window=short_window).mean()
sma_long = daily_crime.rolling(window=long_window).mean()

# Plot all moving averages
plt.figure(figsize=(15, 10))
plt.plot(daily_crime, label='Daily Reported Incidents', color='blue', alpha=0.5)
plt.plot(sma, label='Simple Moving Average (SMA)', color='orange')
plt.plot(ema, label='Exponential Moving Average (EMA)', color='green')
plt.plot(wma, label='Weighted Moving Average (WMA)', color='red')
plt.plot(hma, label='Hull Moving Average (HMA)', color='purple')

# Moving Average Crossover
plt.plot(sma_short, label='SMA (Short Window)', linestyle='--', color='cyan')
plt.plot(sma_long, label='SMA (Long Window)', linestyle='--', color='magenta')

plt.title('Moving Averages and Moving Average Crossover')
plt.xlabel('Date')
plt.ylabel('Reported Incidents')
plt.legend()
plt.grid()
plt.show()
