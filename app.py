import streamlit as st
import pandas as pd
import numpy as np 

# x = st.slider('x')

'''
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)
'''

# num = st.sidebar.slider('Select a number', 0.01, 1, 10)
'''float_num = st.sidebar.slider(
    'Select a float number',  # Label
    min_value=0.0,            # Minimum value
    max_value=1.0,            # Maximum value
    value=0.5,               # Default value
    step=0.01                # Step size
)'''

input_method = st.radio(
    "Choose input method:",
    ('Slider', 'Manual Input')
)

if input_method == 'Slider':
    # Slider for selecting float
    value = st.slider(
        'Select a float number',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
else:
    # Manual input for selecting float
    value = st.number_input(
        'Enter a float number',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

st.write(f'You selected: {value}')


import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Seaborn Heatmap with Interactive Controls")

# Generate Sample Data for the Heatmap
st.sidebar.header("Heatmap Settings")
rows = st.sidebar.slider("Number of Rows", min_value=5, max_value=20, value=10, step=1)
cols = st.sidebar.slider("Number of Columns", min_value=5, max_value=20, value=10, step=1)
cmap = st.sidebar.selectbox("Colormap", ["viridis", "coolwarm", "magma", "plasma", "YlGnBu"])

# Generate Random Data
data = np.random.rand(rows, cols)
df = pd.DataFrame(data, columns=[f"Col_{i+1}" for i in range(cols)])

# Display the DataFrame
st.write("### Dataset for Heatmap")
st.write(df)

# Plot Heatmap
st.write("### Heatmap Visualization")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, ax=ax)
plt.title("Heatmap with Seaborn")

# Show the Heatmap in Streamlit
st.pyplot(fig)
