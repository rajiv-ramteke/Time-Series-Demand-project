
import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly

st.set_page_config(page_title="Walmart Demand Forecast", layout="wide")

st.title("Walmart Demand Forecast Dashboard")

# Load model
model = joblib.load("walmart_demand_model.pkl")

# Load dataset
df = pd.read_csv("Walmart.csv")

df['Date'] = pd.to_datetime(df['Date'])

forecast_df = df[['Date', 'Weekly_Sales']]
forecast_df.columns = ['ds', 'y']

# Sidebar
st.sidebar.header("Forecast Settings")

days = st.sidebar.slider(
    "Select Forecast Days",
    7,
    365,
    90
)

# Future dataframe
future = model.make_future_dataframe(periods=days)

# Prediction
forecast = model.predict(future)

# KPI
st.subheader("Current Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${df['Weekly_Sales'].sum():,.0f}")

col2.metric("Average Sales", f"${df['Weekly_Sales'].mean():,.0f}")

col3.metric("Maximum Sales", f"${df['Weekly_Sales'].max():,.0f}")

# Forecast Chart
st.subheader("Demand Forecast")

fig1 = plot_plotly(model, forecast)

st.plotly_chart(fig1, use_container_width=True)

# Sales Trend
st.subheader("Weekly Sales Trend")

st.line_chart(df['Weekly_Sales'])

# Demand Distribution
st.subheader("Demand Distribution")

st.bar_chart(df['Weekly_Sales'])

# Forecast Table
st.subheader("Forecast Data")

st.write(
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
)

# Raw Data
if st.checkbox("Show Raw Dataset"):
    st.write(df)
