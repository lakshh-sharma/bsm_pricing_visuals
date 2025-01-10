import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optionsPricing as op
import matplotlib.colors as mcolors


# Page Layout
st.set_page_config(layout="wide")
st.title("Black-Scholes-Merton Calculator w/ Heatmaps")
st.markdown("""
You can visualize **3 aspects** from this Black-Scholes Model:

**1. Call & Put Prices with given Inputs**  
**2. Call & Put Prices Heatmap with varying Volatility & Asset Prices**  
**3. Profit & Loss Heatmap after providing Purchase Prices for Call & Put Options**
""")


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

# validating inputs
if min_volatility >= max_volatility:
    st.sidebar.error("Min Volatility must be less than Max Volatility!")
if min_price >= max_price:
    st.sidebar.error("Min Asset Price must be less than Max Asset Price!")

st.sidebar.divider()

# inputs for p&l heatmaps
st.sidebar.header("Purchase Price Input")
call_purchase_price = st.sidebar.number_input("Purchase Price for Call Option", min_value=0.0, value=0.0, step=0.01, key="call_purchase_price")
put_purchase_price = st.sidebar.number_input("Purchase Price for Put Option", min_value=0.0, value=0.0, step=0.01, key="put_purchase_price")

# Heatmap setup using linspace
volatility_range = np.linspace(min_volatility, max_volatility, steps)
price_range = np.linspace(min_price, max_price, steps)

# function to populate the heatmap with the inputs from the sidebar
def generate_heatmap_data(prices, volatilities, k, t, r, calc_function):
    data = np.zeros((len(volatilities), len(prices)))
    for i, vol in enumerate(volatilities):
        for j, price in enumerate(prices):
            data[i, j] = calc_function(price, k, t, r, vol)
    return data

# Heatmap Data for Call and Put
heatmap_data_call = generate_heatmap_data(price_range, volatility_range, k, t, r, op.bsm_call_value)
heatmap_data_put = generate_heatmap_data(price_range, volatility_range, k, t, r, op.bsm_put_value)

# Heatmap Data for P&L
heatmap_data_call_pnl = heatmap_data_call - call_purchase_price
heatmap_data_put_pnl = heatmap_data_put - put_purchase_price

# creating the dataframes to create the heatmpas later using seaborn
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
heatmap_df_call_pnl = pd.DataFrame(
    heatmap_data_call_pnl,
    index=[f"{v:.2f}" for v in volatility_range],
    columns=[f"${p:.2f}" for p in price_range],
)
heatmap_df_put_pnl = pd.DataFrame(
    heatmap_data_put_pnl,
    index=[f"{v:.2f}" for v in volatility_range],
    columns=[f"${p:.2f}" for p in price_range],
)

# colour map for P&L heatmap
cmap_pnl = sns.diverging_palette(150, 10, as_cmap=True)

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


#download buttons for the heatmaps
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

# Coluormap for p&l heatmap
cmap_pnl = mcolors.LinearSegmentedColormap.from_list(
    "pnl_colormap", ["red", "white", "green"], N=100
)
# P&L heatmap visualization
st.write("### Profit & Loss (P&L) Heatmap")
col3, col4 = st.columns(2)

# Call p&l heatmap
with col3:
    st.write("#### Call Option P&L Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_df_call_pnl,
        annot=True,
        fmt=".2f",
        cmap=cmap_pnl,
        cbar_kws={'label': 'P&L Value'},
        ax=ax2,
        annot_kws={"size": 8},
        center=0,  # Center the colormap at zero
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    plt.title("Call Option P&L")
    plt.xlabel("Asset Price")
    plt.ylabel("Volatility")
    st.pyplot(fig2)

# Put p&l heatmap
with col4:
    st.write("#### Put Option P&L Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_df_put_pnl,
        annot=True,
        fmt=".2f",
        cmap=cmap_pnl,
        cbar_kws={'label': 'P&L Value'},
        ax=ax3,
        annot_kws={"size": 8},
        center=0,  # Center the colormap at zero
    )
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)
    plt.title("Put Option P&L")
    plt.xlabel("Asset Price")
    plt.ylabel("Volatility")
    st.pyplot(fig3)


# Download for p&l
st.download_button(
    label="Download Call P&L Heatmap as CSV",
    data=heatmap_df_call_pnl.to_csv(),
    file_name="call_pnl_heatmap.csv",
    mime="text/csv",
)
st.download_button(
    label="Download Put P&L Heatmap as CSV",
    data=heatmap_df_put_pnl.to_csv(),
    file_name="put_pnl_heatmap.csv",
    mime="text/csv",
)
