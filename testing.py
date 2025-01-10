import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optionsPricing as op

# Page Layout
st.set_page_config(layout="wide")
st.title("Black-Scholes-Merton Calculator w/ Heatmap")

# Sidebar
st.sidebar.header("CALL/PUT Option Parameters")

S0 = st.sidebar.number_input('Stock Price', min_value=0.0, max_value=1000000.0, value=1.0, step=1.0, key="Stock Price")
k = st.sidebar.number_input('Strike Price', min_value=0.0, max_value=1000000.0, value=1.0, step=1.0, key="Strike Price")
t = st.sidebar.number_input('Time to Maturity', min_value=0.0, max_value=1000.0, value=1.0, step=1.0, key="time")
vol = st.sidebar.number_input('Volatility', min_value=0.01, max_value=1.0, value=0.5, step=0.01, key="volatility")
r = st.sidebar.number_input('Risk-Free Interest', min_value=0.01, max_value=1.0, value=0.5, step=0.01, key="risk free interest")

st.sidebar.divider()

st.sidebar.header("Heatmap Parameters")
min_volatility = st.sidebar.slider("Min Volatility", 0.01, 1.00, 0.30, step=0.01, key="min_volatility")
max_volatility = st.sidebar.slider("Max Volatility", 0.01, 1.00, 0.50, step=0.01, key="max_volatility")
min_price = st.sidebar.number_input("Min Asset Price", value=50.0, step=1.0, key="min_asset_price")
max_price = st.sidebar.number_input("Max Asset Price", value=150.0, step=1.0, key="max_asset_price")
steps = st.sidebar.slider("Number of Steps", min_value=5, max_value=13, value=10, step=1, key="steps")

if min_volatility >= max_volatility:
    st.sidebar.error("Min Volatility must be less than Max Volatility!")
if min_price >= max_price:
    st.sidebar.error("Min Asset Price must be less than Max Asset Price!")

# New Section: User Input for Purchase Prices
st.sidebar.header("Purchase Price Input")
call_purchase_price = st.sidebar.number_input("Purchase Price for Call Option", min_value=0.0, value=0.0, step=0.01, key="call_purchase_price")
put_purchase_price = st.sidebar.number_input("Purchase Price for Put Option", min_value=0.0, value=0.0, step=0.01, key="put_purchase_price")

# Main Dashboard
st.write("### Input Parameters")
data = {
    "Stock Price (S0)": [S0],
    "Strike Price (K)": [k],
    "Time to Maturity (T)": [t],
    "Volatility (σ)": [vol],
    "Risk-Free Interest (r)": [r]
}
disp = pd.DataFrame(data)
st.table(disp)

# Call and Put Price Calculations
col1, col2 = st.columns(2)
with col1:
    c = op.bsm_call_value(S0, k, t, r, vol)
    call_pnl = c - call_purchase_price  # P&L Calculation for Call
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <div style="
                background-color: #28a745;
                color: white;
                border: 2px solid #218838;
                border-radius: 5px;
                padding: 10px;
                width: 300px;
                text-align: center;
                font-size: 18px;">
                <strong>Call Price</strong><br>${c:.2f}<br>
                <strong>P&L</strong>: ${call_pnl:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    p = op.bsm_put_value(S0, k, t, r, vol)
    put_pnl = p - put_purchase_price  # P&L Calculation for Put
    st.markdown(
        f"""
         <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <div style="
                background-color: #dc3545;
                color: white;
                border: 2px solid #c82333;
                border-radius: 5px;
                padding: 10px;
                width: 300px;
                text-align: center;
                font-size: 18px;">
                <strong>Put Price</strong><br>${p:.2f}<br>
                <strong>P&L</strong>: ${put_pnl:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Heatmap setup
volatility_range = np.linspace(min_volatility, max_volatility, steps)
price_range = np.linspace(min_price, max_price, steps)

def generate_heatmap_data(prices, volatilities, k, t, r, calc_function):
    data = np.zeros((len(volatilities), len(prices)))
    for i, vol in enumerate(volatilities):
        for j, price in enumerate(prices):
            data[i, j] = calc_function(price, k, t, r, vol)
    return data

# Generate Heatmap Data for Call and Put
heatmap_data_call = generate_heatmap_data(price_range, volatility_range, k, t, r, op.bsm_call_value)
heatmap_data_put = generate_heatmap_data(price_range, volatility_range, k, t, r, op.bsm_put_value)

# Convert to DataFrames for Display
heatmap_df_call = pd.DataFrame(
    heatmap_data_call,
    index=[f"{v:.2f}" for v in volatility_range],
    columns=[f"${p:.2f}" for p in price_range],
)
heatmap_df_put = pd.DataFrame(
    heatmap_data_put,
    index=[f"{v:.2f}" for v in volatility_range],
    columns=[f"${p:.2f}" for p in price_range],
)

# Heatmap Visualization
st.write("### Heatmap Visualization")
col1, col2 = st.columns(2)
with col1:
    st.write("#### Call Option Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_df_call,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Option Value'},
        ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title("Call Options")
    plt.xlabel("Asset Price")
    plt.ylabel("Volatility")
    st.pyplot(fig)

with col2:
    st.write("#### Put Option Heatmap")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_df_put,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Option Value'},
        ax=ax1,
        annot_kws={"size": 8}
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    plt.title("Put Options")
    plt.xlabel("Asset Price")
    plt.ylabel("Volatility")
    st.pyplot(fig1)

# Download Buttons
st.download_button(
    label="Download Call Heatmap as CSV",
    data=heatmap_df_call.to_csv(),
    file_name="call_options_heatmap.csv",
    mime="text/csv",
)
st.download_button(
    label="Download Put Heatmap as CSV",
    data=heatmap_df_put.to_csv(),
    file_name="put_options_heatmap.csv",
    mime="text/csv",
)
