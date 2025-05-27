import yfinance as yf
import pandas as pd
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime

symbols = ["AAPL", "MSFT", "TSLA"]  # Replace with your desired symbols

results = []

for symbol in symbols:
    df = yf.download(symbol, period="6mo", interval="1d")
    if df.empty or len(df) < 22:
        continue

    df.dropna(inplace=True)

    # Indicators
    df['EMA9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
    df['EMA9_slope'] = df['EMA9'].diff()
    df['MACD'] = MACD(df['Close']).macd()
    df['Signal'] = MACD(df['Close']).macd_signal()
    df['MACD_Converging'] = (df['MACD'] - df['Signal']).abs().diff().lt(0)
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()

    df['Price_Pos_Days_7'] = df['Close'].pct_change().gt(0).rolling(7).sum()
    df['Price_Pos_Days_10'] = df['Close'].pct_change().gt(0).rolling(10).sum()
    df['Price_Change_10'] = df['Close'].pct_change(10) * 100
    df['Volume_Spike'] = df['Volume'] > df['Volume_MA20']

    # Add weekly price change percentage
    df['Weekly_Price_Change_%'] = df['Close'].pct_change(5) * 100

    # Local Minimum (Buy Point)
    df['Local_Min'] = (
        (df['EMA9_slope'].shift(1) < 0) & (df['EMA9_slope'] >= 0) &
        (df['RSI'] < 40) &
        (df['MACD'] < df['Signal']) &
        df['MACD_Converging'] &
        df['Volume_Spike'] 
    )

    # Local Maximum (Sell Point)
    df['Local_Max'] = (
        (df['EMA9_slope'].shift(1) > 0) & (df['EMA9_slope'] <= 0) &
        (df['RSI'] > 70) &
        (df['MACD'] < df['Signal']) &
        df['Volume_Spike']
    )

    # Breakout Point (Bullish)
    df['Breakout'] = (
        (df['Price_Pos_Days_10'] >= 7) &
        (df['Price_Change_10'] > 2) &
        df['Volume_Spike'] &
        (df['MACD'] > df['Signal']) &
        (df['RSI'].diff() > 0) & (df['RSI'] < 70)
    )

    # Breakdown Point (Bearish)
    df['Breakdown'] = (
        (df['Close'].pct_change().lt(0).rolling(10).sum() >= 5) &
        (df['Price_Change_10'] < 2) &
        df['Volume_Spike'] &
        (df['MACD'] < df['Signal']) &
        (df['RSI'].diff() < 0) & (df['RSI'] < 50)
    )

    for i in range(len(df)):
        if df.iloc[i]['Local_Min']:
            results.append((symbol, 'Local Min', f"{df.iloc[i]['Weekly_Price_Change_%']:.2f}%"))
        elif df.iloc[i]['Local_Max']:
            results.append((symbol, 'Local Max', f"{df.iloc[i]['Weekly_Price_Change_%']:.2f}%"))
        elif df.iloc[i]['Breakout']:
            results.append((symbol, 'Breakout', f"{df.iloc[i]['Weekly_Price_Change_%']:.2f}%"))
        elif df.iloc[i]['Breakdown']:
            results.append((symbol, 'Breakdown', f"{df.iloc[i]['Weekly_Price_Change_%']:.2f}%"))

# Save the result
current_date = datetime.now().strftime("%Y-%m-%d")
output_file = f"price_points_{current_date}.txt"
with open(output_file, 'w') as f:
    f.write("Symbol | Signal Type | Weekly % Change\n")
    f.write("-" * 50 + "\n")
    for row in results:
        f.write(" | ".join(row) + "\n")

print(f"Price movement points saved to {output_file}")
