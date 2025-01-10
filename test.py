import optionsPricing
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Seaborn Heatmap with Interactive Controls")

# Generate Sample Data for the Heatmap
st.sidebar.header("Heatmap Settings")
S0 = st.sidebar.number_input('Stock Price', min_value=0.0, max_value=1000000.0, value=1.0, step=1.0, key = "Stock Price")
k = st.sidebar.number_input('Strike Price', min_value=0.0, max_value=1000000.0, value=1.0, step=1.0, key = "Strike Price")
t = st.sidebar.number_input('Time to Maturity', min_value=0.0, max_value=1000.0, value=1.0, step=1.0, key = "time")
vol = st.sidebar.number_input('Volatility', min_value=0.01, max_value=1.0, value=0.5, step=0.01, key = "volatility")
r = st.sidebar.number_input('Risk Free Interest', min_value=0.01, max_value=1.0, value=0.5, step=0.01, key = "risk free interest")
minVol = st.sidebar.slider("Minimum Volatility", min_value = 0.01, max_value = 1.0, value = 0.5, step = 0.01, key = "minVol")
maxVol = st.sidebar.slider("Maximum Volatility", min_value = 0.01, max_value = 1.0, value = 0.5, step = 0.01, key = "maxVol")

cmap = st.sidebar.selectbox("Colormap", ["viridis", "coolwarm", "magma", "plasma", "YlGnBu"])

# Generate Random Data
df = pd.DataFrame()
dict = {}

'''
def generator():
    for i in range(1, 11):

    '''


# Plot Heatmap
st.write("### Heatmap Visualization")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, ax=ax)
plt.title("Call Price Heatmap")

# Show the Heatmap in Streamlit
st.pyplot(fig)