# Stock Price Prediction using TensorFlow

This project implements a stock price prediction model using TensorFlow and LSTM. It analyzes historical stock data to forecast future prices.

## Project Structure

- `data/`: Contains datasets used for training and testing.
- `notebooks/`: Jupyter notebooks detailing data exploration and model development.
- `models/`: Stored trained models.
- `scripts/`: Python scripts for data processing and model operations.
- `requirements.txt`: Python dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```

2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the data processing script:
   ```bash
   python scripts/data_processing.py
   ```

5. Train the model:
   ```bash
   python scripts/train_model.py
   ```

6. Evaluate the model:
   ```bash
   python scripts/evaluate_model.py
   ```

---

## Install Dependencies

Create a `requirements.txt` file:
```plaintext
pandas
numpy
matplotlib
tensorflow
scikit-learn
```

Install using:
```bash
pip install -r requirements.txt
```

---

## Data Processing Script
Create `scripts/data_processing.py`:
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('data/stock_data.csv')

data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

data.to_csv('data/processed_stock_data.csv', index=False)
```

---

## Model Training Script
Create `scripts/train_model.py`:
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load data
data = pd.read_csv('data/processed_stock_data.csv')
seq_length = 50
X, y = create_sequences(data['Close'].values, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)
model.save('models/stock_price_model.h5')
```

---

## Model Evaluation Script
Create `scripts/evaluate_model.py`:
```python
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
```

---

## Push Code to GitHub

```bash
# Initialize Git repository
git init

git add .
git commit -m "Initial commit: Stock Price Prediction Project"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main

---

### ** data/` Directory (Dataset Storage)**
If you don’t have a dataset yet, create a placeholder CSV file:

**`data/stock_data.csv`**
```plaintext
Date,Open,High,Low,Close,Volume
2023-01-01,100,105,99,104,1000000
2023-01-02,104,108,102,107,1200000
# Add more stock price data...
