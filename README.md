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

Exploratory Data Analysis
EDA also known as Exploratory Data Analysis is a technique that is used to analyze the data through visualization and manipulation. For this project let us visualize the data of famous companies such as Nvidia, Google, Apple, Facebook, and so on.

First, let us consider a few companies and visualize the distribution of open and closed Stock prices through 5 years. 


data['date'] = pd.to_datetime(data['date']) 
# date vs open 
# date vs close

# Define the list of companies you want to plot
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

plt.figure(figsize=(15, 8)) 
for index, company in enumerate(companies, 1): 
    plt.subplot(3, 3, index) 
    c = data[data['Name'] == company] 
    plt.plot(c['date'], c['close'], c="r", label="close", marker="+") 
    plt.plot(c['date'], c['open'], c="g", label="open", marker="^") 
    plt.title(company) 
    plt.legend() 
    plt.tight_layout()
    

Output:
![image](https://github.com/user-attachments/assets/721e32c4-35ea-44b6-a015-d8d5111aab2c)
Now let’s plot the volume of trade for these 9 stocks as well as a function of time.


plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
    plt.tight_layout()
Output:
![image](https://github.com/user-attachments/assets/16af9123-869e-4937-8792-27d1e78d1386)
Now let’s analyze the data for Apple Stocks from 2013 to 2018.


apple = data[data['Name'] == 'AAPL']
prediction_range = apple.loc[(apple['date'] > datetime(2013,1,1))
 & (apple['date']<datetime(2018,1,1))]
plt.plot(apple['date'],apple['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Apple Stock Prices")
plt.show()
Output:
![image](https://github.com/user-attachments/assets/a9f4adbb-4f44-42d4-96ae-b4869575cfdb)
Now let’s select a subset of the whole data as the training data so, that we will be left with a subset of the data for the validation part as well.


close_data = apple.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)
Output:
1197
Now we have the training data length, next applying scaling and preparing features and labels that are x_train and y_train. 


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]
# prepare feature and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
Build Gated RNN- LSTM network using TensorFlow
Using TensorFlow, we can easily create LSTM-gated RNN cells. LSTM is used in Recurrent Neural Networks for sequence models and time series data. LSTM is used to avoid the vanishing gradient issue which is widely occurred in training RNN. To stack multiple LSTM in TensorFlow it is mandatory to use return_sequences = True. Since our data is time series varying we apply no activation to the output layer and it remains as 1 node. 


model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary
Output:
![image](https://github.com/user-attachments/assets/2763e15a-a72a-4b15-b11f-198606e1dbb0)
Model Compilation and Training
While compiling a model we provide these three essential parameters:

optimizer – This is the method that helps to optimize the cost function by using gradient descent.
loss – The loss function by which we monitor whether the model is improving with training or not.
metrics – This helps to evaluate the model by predicting the training and the validation data.

model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(x_train,
                    y_train,
                    epochs=10)
Output:
![image](https://github.com/user-attachments/assets/1ef08d1f-6de7-451f-946d-d7f76f0bac23)
For predicting we require testing data, so we first create the testing data and then proceed with the model prediction. 


test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))
Output:

2/2 [==============================] - 1s 13ms/step
MSE 34.42497277619552
RMSE 5.867279844714714
Now that we have predicted the testing data, let us visualize the final results. 


train = apple[:training]
test = apple[training:]
test['Predictions'] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train['date'], train['close'])
plt.plot(test['date'], test[['close', 'Predictions']])
plt.title('Apple Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])

Output:
![image](https://github.com/user-attachments/assets/2f0ff30f-e0a0-4901-9294-456b653f7aa7)



