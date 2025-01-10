import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optionsPricing as op

# Title
st.title("Black-Scholes-Merton Options Heatmap")

# Sidebar Inputs
st.sidebar.header("CALL/PUT Option Parameters")

S0 = st.sidebar.number_input('Stock Price', min_value=0.0, max_value=1000000.0, value=1.0, step=1.0, key = "Stock Price")
k = st.sidebar.number_input('Strike Price', min_value=0.0, max_value=1000000.0, value=1.0, step=1.0, key = "Strike Price")
t = st.sidebar.number_input('Time to Maturity', min_value=0.0, max_value=1000.0, value=1.0, step=1.0, key = "time")
vol = st.sidebar.number_input('Volatility', min_value=0.01, max_value=1.0, value=0.5, step=0.01, key = "volatility")
r = st.sidebar.number_input('Risk Free Interest', min_value=0.01, max_value=1.0, value=0.5, step=0.01, key = "risk free interest")

st.sidebar.divider()

st.sidebar.header("Heatmap Parameters")


st.write("### Input Parameters")

# Method 1: Using st.table with a DataFrame
data = {
    "Stock Price (S0)": [S0],
    "Strike Price (K)": [k],
    "Time to Maturity (T)": [t],
    "Volatility (σ)": [vol],
    "Risk-Free Interest (r)": [r]
}
disp = pd.DataFrame(data)
st.table(disp)



col1, col2 = st.columns(2)

with col1:
    c = op.bsm_call_value(S0, k, t, r, vol)
    st.markdown(
        f"""
        <div style="
            background-color: #28a745;
            color: white;
            border: 2px solid #218838;
            border-radius: 5px;
            padding: 10px;
            width: 300px;
            text-align: center;
            font-size: 18px;">
            <strong>Call Price</strong><br>${c:.2f}
        </div>
        """,
        unsafe_allow_html=True,
        
    )

    

with col2:
    p = op.bsm_put_value(S0, k, t, r, vol)
    st.markdown(
        f"""
        <div style="
            background-color: #dc3545;
            color: white;
            border: 2px solid #c82333;
            border-radius: 5px;
            padding: 10px;
            width: 300px;
            text-align: center;
            font-size: 18px;">
            <strong>Put Price</strong><br>${p:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )



# Sliders for Volatility
min_volatility = st.sidebar.slider("Min Volatility", 0.01, 1.00, 0.50, step=0.01, key = "min_volatility")
max_volatility = st.sidebar.slider("Max Volatility", 0.01, 1.00, 0.50, step=0.01, key = "max_volatility")

# Inputs for Asset Price
min_price = st.sidebar.number_input("Min Asset Price", value=50.0, step=1.0, key = "min_asset_price")
max_price = st.sidebar.number_input("Max Asset Price", value=150.0, step=1.0, key = "max_asset_price")

# Number of Steps
steps = st.sidebar.slider("Number of Steps", min_value=5, max_value=15, value=10, step=1, key = "steps")

# Generate Ranges for Volatility and Asset Price
volatility_range = np.linspace(min_volatility, max_volatility, steps)
price_range = np.linspace(min_price, max_price, steps)

# Create a DataFrame for the Heatmap
heatmap_data_call = np.zeros((steps, steps))

# Populate the Heatmap with Option Values (Dummy Calculation)
for i, vol in enumerate(volatility_range):
    for j, price in enumerate(price_range):
        call = op.bsm_call_value(price, k, t, r, vol)
        heatmap_data_call[i, j] = call # Replace with your calculation logic

# Convert to DataFrame for Labeling
heatmap_df_call = pd.DataFrame(
    heatmap_data_call,
    index=[f"{v:.1f}" for v in volatility_range],  # Row labels: Volatility
    columns=[f"${p:.1f}" for p in price_range],    # Column labels: Asset Price
)

heatmap_data_put = np.zeros((steps, steps))

# Populate the Heatmap with Option Values (Dummy Calculation)
for i, vol in enumerate(volatility_range):
    for j, price in enumerate(price_range):
        call = op.bsm_put_value(price, k, t, r, vol)
        heatmap_data_put[i, j] = call # Replace with your calculation logic

# Convert to DataFrame for Labeling
heatmap_df_put = pd.DataFrame(
    heatmap_data_put,
    index=[f"{v:.1f}" for v in volatility_range],  # Row labels: Volatility
    columns=[f"${p:.1f}" for p in price_range],    # Column labels: Asset Price
)

# Display the Heatmap
st.write("### Heatmap Visualization")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_df_call, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Option Value'}, ax=ax)
plt.title("Call Options")
plt.xlabel("Asset Price")
plt.ylabel("Volatility")

fig1, ax1 = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_df_put, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Option Value'}, ax=ax1)
plt.title("Put Options")
plt.xlabel("Asset Price")
plt.ylabel("Volatility")

# Show Heatmap in Streamlit
st.pyplot(fig)
st.pyplot(fig1)

