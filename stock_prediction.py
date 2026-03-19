import yfinance as yf
data = yf.download("AAPL", start="2025-01-01", end="2026-01-01")
print(data.head())

#data preprosessing
print(data.isnull().sum())

#feature engineering
data = data[['Close']]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])
print(data.head())

#visulization
import matplotlib.pyplot as plt

data['Close'].plot(figsize=(10,5))
plt.title("Stock Price Visualization")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# convert to array
dataset = data.values

# training data length (80%)
training_data_len = int(len(dataset) * 0.8)

# training data
train_data = dataset[0:training_data_len]

# create X and y
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i])  # last 60 days
    y_train.append(train_data[i])       # next day

# convert to numpy arrays
import numpy as np
x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)
print(y_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=20, batch_size=32)

test_data = dataset[training_data_len - 60:]
x_test = []
y_test = dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i])

x_test = np.array(x_test)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R² Score:", r2)

plt.figure(figsize=(12,6))
plt.plot(y_test[-100:], label='Actual Price')
plt.plot(predictions[-100:], label='Predicted Price')
plt.title("AAPL Stock Price Prediction (Last 100 Days)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# Reshape training and test data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

from tensorflow.keras.models import load_model

# Save model
model.save("aapl_lstm_model.h5")

# Later, you can load it
# model = load_model("aapl_lstm_model.h5")

last_60_days = dataset[-60:]
last_60_days_scaled = scaler.transform(last_60_days.reshape(-1,1))

X_input = np.array([last_60_days_scaled])
X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 1))

next_day_pred = model.predict(X_input)
next_day_price = scaler.inverse_transform(next_day_pred)
print("Next Day Predicted Price:", next_day_price[0][0])

plt.figure(figsize=(12,6))
plt.plot(y_test[-100:], label='Actual Price')
plt.plot(predictions[-100:], label='Predicted Price')
plt.title("AAPL Stock Price Prediction (Last 100 Days)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

#predict multiple days model
# keep everything in scaled form
future_predictions = []
last_60_days = dataset[-60:]  # already scaled

for i in range(5):
    X_input = np.array([last_60_days]).reshape(1, 60, 1)
    
    pred_scaled = model.predict(X_input)
    
    # store actual value
    pred_actual = scaler.inverse_transform(pred_scaled)[0][0]
    future_predictions.append(pred_actual)
    
    # append SCALED value (important)
    last_60_days = np.append(last_60_days[1:], pred_scaled, axis=0)


    

# convert all predictions to float and print
future_prices = [float(x) for x in future_predictions]
print("Next 5 Days Predicted Prices:", future_prices)

# Plot actual, predicted, and future 5-day predictions
plt.figure(figsize=(12,6))
plt.plot(list(y_test.flatten()), label='Actual Price')
plt.plot(list(predictions.flatten()), label='Predicted Price')
plt.plot(
    range(len(y_test), len(y_test) + len(future_prices)),
    future_prices,
    label='Future 5 Days Prediction',
    marker='o',
    color='red'
)
plt.title("AAPL Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# Save the model
model.save("aapl_lstm_model.h5")


