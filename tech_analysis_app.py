import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from sector_mapping import sector_stocks

# Set Streamlit layout
st.set_page_config(layout="wide")

# Sidebar - Sector and stock selection
with st.sidebar:
    st.header("Select Sector and Stock")
    selected_sector = st.selectbox("Sector", list(sector_stocks.keys()))
    stocks = sector_stocks[selected_sector]
    selected_stock = st.radio("Stocks", stocks)

    st.markdown("---")
    st.header("Search Stock Directly")
    direct_stock = st.text_input("Enter stock symbol (e.g., INFY.NS)")

# Decide which stock to use
symbol_to_use = direct_stock if direct_stock else selected_stock

# Main - Technical Analysis
if symbol_to_use:
    stck = yf.Ticker(symbol_to_use)
    df = stck.history(period="6mo", interval="1d")
    df.dropna(inplace=True)

    # Indicators
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    # Layout with columns
    left_col, right_col = st.columns([1, 8])

    with right_col:
        st.subheader(f"Technical Analysis for {symbol_to_use}")

        # Plot 0: Candlestick Chart
        st.write("### Candlestick Chart")
        df_mpf = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_mpf.index.name = 'Date'
        fig_candle, ax_candle = mpf.plot(
            df_mpf,
            type='candle',
            mav=(20, 50),
            volume=True,
            style='yahoo',
            returnfig=True,
            figsize=(10, 5)
        )        
        st.pyplot(fig_candle)


        # Plot 1: Price + MA + Bollinger
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], label='Close', color='black')
        ax.plot(df.index, df['MA20'], label='MA20', color='blue')
        ax.plot(df.index, df['MA50'], label='MA50', color='orange')
        ax.set_title("Price with MA")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Plot 2: RSI
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax.axhline(70, color='red', linestyle='--')
        ax.axhline(30, color='green', linestyle='--')
        ax.set_title("Relative Strength Index (RSI)")
        ax.grid()
        st.pyplot(fig)

        # Plot 3: MACD
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax.plot(df.index, df['Signal'], label='Signal', color='red')
        ax.set_title("MACD")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Plot 1: Price + MA + Bollinger
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], label='Close', color='black')
        ax.plot(df.index, df['BB_Upper'], linestyle='--', color='red', label='BB Upper')
        ax.plot(df.index, df['BB_Middle'], color='gray', label='BB Middle')
        ax.plot(df.index, df['BB_Lower'], linestyle='--', color='green', label='BB Lower')
        ax.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], color='lightgray', alpha=0.2)
        ax.set_title("Price with Bollinger bands")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        
