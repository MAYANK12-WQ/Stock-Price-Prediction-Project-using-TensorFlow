import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('data/processed_stock_data.csv')

# Load model
model = tf.keras.models.load_model('models/stock_price_model.h5')

# Prepare test data
seq_length = 50
X, y = create_sequences(data['Close'].values, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Predict
predictions = model.predict(X)

# Inverse transform predictions
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(data[['Close']])
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(data['Close'][seq_length:], predictions))
print(f'Root Mean Squared Error: {rmse}')

# Plot results
plt.figure(figsize=(14, 5))
plt.plot(data['Date'][seq_length:], data['Close'][seq_length:], label='Actual Prices')
plt.plot(data['Date'][seq_length:], predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
