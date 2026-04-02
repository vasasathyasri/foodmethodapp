import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

st.title("🍔 Food Demand Forecasting App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data")
    st.write(data.head())

    data['ds'] = pd.to_datetime(data['date'])
    data['y'] = data['orders']

    data['day_of_week'] = data['ds'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['is_holiday'] = data['holiday'].notnull().astype(int)
    data['rain'] = (data['rainfall_mm'] > 0).astype(int)
    data['promotion'] = data['discount_percent'].notnull().astype(int)
    data['new_menu_item'] = data['new_item'].notnull().astype(int)

    if st.button("Train Model 🚀"):
        model = Prophet()
        model.add_regressor('is_weekend')
        model.add_regressor('is_holiday')
        model.add_regressor('rain')
        model.add_regressor('promotion')
        model.add_regressor('new_menu_item')

        model.fit(data)

        future = model.make_future_dataframe(periods=30)
        future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        future['is_holiday'] = 0
        future['rain'] = 0
        future['promotion'] = 0
        future['new_menu_item'] = 0

        forecast = model.predict(future)

        y_true = data['y']
        y_pred = forecast['yhat'][:len(data)]

        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)

        st.subheader("📈 Model Performance")
        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")

        st.subheader("📊 Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("📉 Actual vs Predicted")
        fig2 = plt.figure()
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        st.pyplot(fig2)
