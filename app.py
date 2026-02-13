import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import backend as K


@st.cache_resource
def load_lstm_model():
    from tensorflow.keras.models import load_model
    return load_model("model_fixed.h5", compile=False)


st.title("Cryptocurrency Price Predictor (LSTM)")

# Load model once using Streamlit cache
try:
    model = load_lstm_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Inputs
stock = st.text_input("Enter crypto ticker", "BTC-USD")
days = st.slider("Days to predict", 1, 30, 10)

if st.button("Predict"):
    st.write("Predict button clicked")

    try:
        # Fetch data
        st.write("Downloading data...")
        end = datetime.now()
        start = datetime(end.year - 15, end.month, end.day)
        data = yf.download(stock, start=start, end=end)

        if data.empty:
            st.error("Invalid ticker or no data available.")
            st.stop()

        st.write("Data loaded successfully")

        close = data[['Close']]

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close)
        scaled_data = scaled_data.astype(np.float32)

        base_days = 100

        if len(scaled_data) < base_days:
            st.error("Not enough data for prediction.")
            st.stop()

        # Prepare last window
        last_window = scaled_data[-base_days:].reshape(1, base_days, 1)

        future_predictions = []

        st.write("Running model predictions...")

        last_window = last_window.astype(np.float32)

        with st.spinner("Predicting future prices..."):
            for _ in range(days):
                next_day = model(last_window, training=False).numpy()
                future_price = scaler.inverse_transform(next_day)[0][0]
                future_predictions.append(float(future_price))

                # maintain float32 consistency
                next_day = next_day.astype(np.float32)
                last_window = np.concatenate(
                    [last_window[:, 1:, :], next_day.reshape(1, 1, 1)],
                    axis=1
                )
                last_window = last_window.astype(np.float32)

        st.write("Predictions complete")

        # Plot historical data
        st.subheader("Closing Price History")
        fig1, ax1 = plt.subplots()
        ax1.plot(close.index, close['Close'])
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        st.pyplot(fig1)

        # Plot predictions
        st.subheader("Future Predictions")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, days + 1), future_predictions, marker='o')
        ax2.set_xlabel("Day Ahead")
        ax2.set_ylabel("Predicted Price")
        st.pyplot(fig2)

        # Table
        st.subheader("Prediction Table")
        df = pd.DataFrame({
            "Day": range(1, days + 1),
            "Predicted Price": future_predictions
        })
        st.dataframe(df)

    except Exception as e:
        st.error(f"Prediction error: {e}")
