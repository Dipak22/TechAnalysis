import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
from sector_mapping import sector_stocks
import time
import logging
import numpy as np

# Configure logging
logging.basicConfig(filename='stock_demand_analysis_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Flatten all stocks from all sectors
all_stocks = [stock for stocks in sector_stocks.values() for stock in stocks]

# Initialize results
high_demand_stocks = []
low_demand_stocks = []

def calculate_demand_score(df):
    """Calculate a composite demand score based on multiple technical factors"""
    scores = {}
    
    # 1. Price Momentum (30% weight)
    df['5_day_return'] = df['Close'].pct_change(5)
    df['10_day_return'] = df['Close'].pct_change(10)
    scores['momentum'] = 0.5 * df['5_day_return'].iloc[-1] + 0.5 * df['10_day_return'].iloc[-1]
    
    # 2. Volume Analysis (25% weight)
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    # Replace the volume ratio line with:
    volume_ratio = (df['Volume'].iloc[-1] + 1e-6) / (df['Volume_MA_20'].iloc[-1] + 1e-6)
    scores['volume'] = np.log(volume_ratio)
    
    # 3. Positive Days Ratio (15% weight)
    df['Pct_Change'] = df['Close'].pct_change()
    last_10 = df['Pct_Change'].tail(10)
    pos_days = (last_10 > 0).sum()
    scores['positive_days'] = pos_days / 10.0
    
    # 4. RSI Indicator (15% weight)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    scores['rsi'] = (df['RSI'].iloc[-1] - 50) / 50  # Normalize to -1 to 1 range
    
    # 5. Price-Volume Trend (15% weight)
    df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
    pvt_change = df['PVT'].iloc[-1] - df['PVT'].iloc[-5]
    scores['pvt'] = pvt_change / df['Volume'].mean()
    
    # Calculate composite score (weighted average)
    weights = {
        'momentum': 0.20,
        'volume': 0.05,
        'positive_days': 0.60,
        'rsi': 0.05,
        'pvt': 0.10
    }
    for key in scores:
        print(scores[key])
    composite_score = sum(scores[key] * weights[key] for key in scores)
    return composite_score

for symbol in ["PUNJABCHEM.NS","PRIVISCL.NS","RPGLIFE.NS","CURAA.NS","HECPROJECT.NS","ORTEL.NS","ARSSINFRA.NS"]:
    success = False
    for attempt in range(3):  # Try up to 3 times
        #time.sleep(1)  # Sleep for 1 second between requests
        if success:
            break
        try:
            # Get data - using 3 months for more reliable indicators
            end_date = datetime.today()
            start_date = end_date - timedelta(days=90)
            stck = yf.Ticker(symbol)
            df = stck.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 20:  # Need at least 20 days for indicators
                continue
                
            df.dropna(inplace=True)
            print(symbol)
            # Calculate demand score
            demand_score = calculate_demand_score(df)
            
            # Classify based on score
            if demand_score > 0.5:  # Strong demand threshold
                high_demand_stocks.append((symbol, demand_score))
            elif demand_score < -0.5:  # Weak demand threshold
                low_demand_stocks.append((symbol, demand_score))
                
            success = True
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt == 2:
                logging.error(f"Failed to process {symbol} after 3 attempts")

# Sort stocks by demand score
high_demand_stocks.sort(key=lambda x: x[1], reverse=True)
low_demand_stocks.sort(key=lambda x: x[1])

# Save to file
current_date = datetime.now().strftime("%Y-%m-%d")
filename = f"stock_demand_analysis_{current_date}.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"Stock Demand Analysis Report - {current_date}\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("HIGH DEMAND STOCKS (Strong Buying Pressure):\n")
    f.write("-" * 50 + "\n")
    for stock, score in high_demand_stocks:
        f.write(f"{stock.ljust(15)} Score: {score:.2f}\n")
    
    f.write("\n\nLOW DEMAND STOCKS (Selling Pressure/Weak Interest):\n")
    f.write("-" * 50 + "\n")
    for stock, score in low_demand_stocks:
        f.write(f"{stock.ljust(15)} Score: {score:.2f}\n")

print(f"Analysis saved to {filename}")
print(f"Found {len(high_demand_stocks)} high demand stocks and {len(low_demand_stocks)} low demand stocks")