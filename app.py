import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Title
st.title("📈 Stock Price Prediction App")

# User input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")

# Load model
model = load_model("aapl_lstm_model.h5")

# Download data
data = yf.download(stock, start="2025-01-01", end="2026-01-01")

# Fix multi-index issue
data.columns = data.columns.get_level_values(0)

# Check data
if len(data) < 60:
    st.error("❌ Not enough data to predict. Try another stock.")
    st.stop()

# Use Close price
data = data[['Close']]

# Show raw data
st.subheader("Raw Data")
st.write(data.tail())

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare last 60 days
last_60_days = scaled_data[-60:]

# Reshape for model
X_input = last_60_days.reshape(1, last_60_days.shape[0], 1)

# Predict next day
predicted_price = model.predict(X_input)
predicted_price = scaler.inverse_transform(predicted_price)

st.subheader("📊 Next Day Prediction")
st.success(f"Predicted Price: ${predicted_price[0][0]:.2f}")

# Predict next 5 days
future_predictions = []
temp_input = last_60_days.copy()

for i in range(5):
    X_input = temp_input.reshape(1, 60, 1)
    pred = model.predict(X_input)
    
    future_predictions.append(scaler.inverse_transform(pred)[0][0])
    
    # update input
    temp_input = np.append(temp_input[1:], pred, axis=0)

# Show future predictions
st.subheader("📅 Next 5 Days Prediction")
for i, price in enumerate(future_predictions):
    st.write(f"Day {i+1}: ${price:.2f}")

# Plot graph
st.subheader("📉 Stock Price Graph")

plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Actual Price')
plt.title(f"{stock} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

st.pyplot(plt)