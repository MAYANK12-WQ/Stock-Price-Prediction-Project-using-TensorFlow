import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load processed data
data = pd.read_csv('data/processed_stock_data.csv')

# Load model
model = tf.keras.models.load_model('models/stock_price_model.h5')

# Prepare test data
seq_length = 50
X, y = create_sequences(data['Close'].values, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Predict prices
predictions = model.predict(X)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(data['Close'][seq_length:], predictions))
print(f'📊 Root Mean Squared Error: {rmse}')

# Plot predictions vs actual prices
plt.figure(figsize=(14, 5))
plt.plot(data['Date'][seq_length:], data['Close'][seq_length:], label='Actual Prices')
plt.plot(data['Date'][seq_length:], predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
