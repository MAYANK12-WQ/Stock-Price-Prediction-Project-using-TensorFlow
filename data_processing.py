import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('data/stock_data.csv')

# Select relevant columns
data = data[['Date', 'Close']]

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date
data = data.sort_values('Date')

# Normalize 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

# Save processed data
data.to_csv('data/processed_stock_data.csv', index=False)
