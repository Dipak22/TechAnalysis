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
    # Fibonacci Levels
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price
    df['FIB_0'] = max_price
    df['FIB_236'] = max_price - 0.236 * diff
    df['FIB_382'] = max_price - 0.382 * diff
    df['FIB_5'] = max_price - 0.5 * diff
    df['FIB_618'] = max_price - 0.618 * diff
    df['FIB_786'] = max_price - 0.786 * diff
    df['FIB_100'] = min_price
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['Daily_StdDev'] = df['Close'].rolling(window=2).std()
    df['MA09'] = ta.trend.sma_indicator(df['Close'], window=9)
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=9)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # Add Parabolic SAR
    df['PSAR'] = ta.trend.PSARIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        step=0.02,  # standard value
        max_step=0.2  # standard value
    ).psar()
    
    # Add ADX (Average Directional Index)
    adx_indicator = ta.trend.ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14  # standard period
    )
    df['ADX'] = adx_indicator.adx()
    df['DMP'] = adx_indicator.adx_pos()  # +DI
    df['DMN'] = adx_indicator.adx_neg()  # -DI

    # Layout with columns
    left_col, right_col = st.columns([1, 8])

    with right_col:
        st.subheader(f"Technical Analysis for {symbol_to_use}")

         # Plot 0: Candlestick Chart
        st.write("### Candlestick Chart")
        df_mpf = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_mpf.index.name = 'Date'
        addplot = [
            mpf.make_addplot(df['VWAP'], color='magenta'),
            mpf.make_addplot(df['PSAR'], type='scatter', markersize=50, marker='.', color='blue')
        ]
        fig_candle, ax_candle = mpf.plot(
            df_mpf,
            type='candle',
            mav=(20, 50),
            volume=True,
            style='yahoo',
            addplot=addplot,
            returnfig=True,
            figsize=(10, 5),
            update_width_config=dict(candle_linewidth=1.0)
        )
        # Add legend manually since mplfinance does not auto-legend custom addplot
        ax_candle[0].legend([
            "MA20 (Blue)", 
            "MA50 (Orange)", 
            "VWAP (Magenta)",
            "PSAR (Blue dots)"
        ], loc='upper left')
        st.pyplot(fig_candle)

        # Plot 1: Price + MA
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], label='Close', color='black')
        ax.plot(df.index, df['MA20'], label='MA20', color='blue')
        ax.plot(df.index, df['MA50'], label='MA50', color='orange')
        ax.plot(df.index, df['MA09'], label='MA9', color='green')
        ax.set_title("Price with MA")
        ax.legend()
                # Plot Fibonacci Levels
        ax.axhline(df['FIB_0'].iloc[-1], linestyle='--', color='violet', alpha=0.7, label='FIB 0%')
        ax.axhline(df['FIB_236'].iloc[-1], linestyle='--', color='indigo', alpha=0.7, label='FIB 23.6%')
        ax.axhline(df['FIB_382'].iloc[-1], linestyle='--', color='blue', alpha=0.7, label='FIB 38.2%')
        ax.axhline(df['FIB_5'].iloc[-1], linestyle='--', color='cyan', alpha=0.7, label='FIB 50%')
        ax.axhline(df['FIB_618'].iloc[-1], linestyle='--', color='green', alpha=0.7, label='FIB 61.8%')
        ax.axhline(df['FIB_786'].iloc[-1], linestyle='--', color='orange', alpha=0.7, label='FIB 78.6%')
        ax.axhline(df['FIB_100'].iloc[-1], linestyle='--', color='red', alpha=0.7, label='FIB 100%')
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

        # Plot 4: Price + Bollinger
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
        
        # Plot 5: ADX with +DI and -DI
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df.index, df['ADX'], label='ADX', color='black', linewidth=2)
        ax.plot(df.index, df['DMP'], label='+DI', color='green')
        ax.plot(df.index, df['DMN'], label='-DI', color='red')
        ax.axhline(25, color='gray', linestyle='--', alpha=0.7)
        ax.set_title("ADX with Directional Indicators (Trend Strength)")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        
        # Interpretation guide
        st.markdown("""
        ### Trend Identification Guide:
        - **Parabolic SAR (PSAR)**: 
          - Dots below price → Uptrend
          - Dots above price → Downtrend
          - The closer the dots to price, the stronger the trend
        - **ADX (Average Directional Index)**:
          - ADX > 25 → Strong trend
          - ADX < 20 → Weak trend/range-bound
          - +DI > -DI → Bullish trend
          - -DI > +DI → Bearish trend
        """)