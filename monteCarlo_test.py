import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

# 1. DEFINE PARAMETERS
ticker = 'AAPL'  # Example: Apple stock
option_expiry = '2024-03-15'  # Choose an option expiry date
risk_free_rate = 0.04  # Assume 4% annual risk-free rate (you can fetch from FRED API)

# 2. FETCH STOCK DATA
stock = yf.Ticker(ticker)

# Current Stock Price (S0)
current_price = stock.history(period='1d')['Close'].iloc[-1]
print(f"Current Price: {current_price}")

# Fetch Historical Stock Data for Volatility Calculation
historical_data = stock.history(period='1y')
historical_volatility = historical_data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
print(f"Historical Volatility: {historical_volatility:.2f}")

# 3. FETCH OPTION CHAIN DATA
# Fetch Available Expiration Dates
expirations = stock.options
print("Available Expiration Dates:", expirations)

# Select Option Chain for a Specific Expiry Date
if option_expiry in expirations:
    option_chain = stock.option_chain(option_expiry)
    
    # Choose Closest At-The-Money (ATM) Strike Price
    atm_call = option_chain.calls.loc[
        (option_chain.calls['strike'] >= current_price)
    ].iloc[0]
    
    strike_price = atm_call['strike']
    implied_volatility = atm_call['impliedVolatility']
    
    print(f"Strike Price: {strike_price}")
    print(f"Implied Volatility: {implied_volatility:.2f}")
else:
    print(f"Option expiry {option_expiry} not found. Please choose from the available dates.")
    exit()

# 4. TIME TO MATURITY (T)
today = dt.datetime.now()
expiry_date = dt.datetime.strptime(option_expiry, '%Y-%m-%d')
time_to_maturity = (expiry_date - today).days / 365  # Time in years
print(f"Time to Maturity: {time_to_maturity:.2f} years")

# 5. DISPLAY FETCHED DATA
option_params = {
    'Stock Ticker': ticker,
    'Current Stock Price (S0)': current_price,
    'Strike Price (K)': strike_price,
    'Time to Maturity (T)': time_to_maturity,
    'Historical Volatility (σ)': historical_volatility,
    'Implied Volatility': implied_volatility,
    'Risk-Free Rate (r)': risk_free_rate
}

df = pd.DataFrame(option_params, index=[0])
print("\n📊 Fetched Parameters for Monte Carlo Simulation:")
print(df)
