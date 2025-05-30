import yfinance as yf
import pandas as pd
import ta
from datetime import datetime
from sector_mapping import sector_stocks
import time
import logging
from ta.trend import MACD, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

# Configure logging
logging.basicConfig(filename='macd_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Flatten all stocks from all sectors
all_stocks = [stock for stocks in sector_stocks.values() for stock in stocks]
#all_stocks = [ "NEWGEN.NS","MAPMYINDIA.NS","INTELLECT.NS","INSPIRISYS.NS"]

buy_signals = []
sell_signals = []

for symbol in all_stocks:
    success = False
    for attempt in range(3):  # Try up to 3 times
        time.sleep(1)  # Sleep for 1 second between requests
        if success:
            break
        try:
            stck = yf.Ticker(symbol)
            df = stck.history(period="6mo", interval="1d")
            df.dropna(inplace=True)
            #df = yf.download(symbol, period="6mo", interval="1d")
            if df.empty:
                continue
            df.dropna(inplace=True)

            # Indicators
            df['EMA9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
            df['MACD'] = MACD(df['Close']).macd()
            df['Signal'] = MACD(df['Close']).macd_signal()
            vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
            df['VWAP'] = vwap.volume_weighted_average_price()

            # Buy/Sell Conditions
            df['Buy_Signal'] = (df['Close'] < df['VWAP']) & (df['Close'] < df['EMA9']) & (df['MACD'] < df['Signal'])#(df['Close'] > df['VWAP']) & (df['Close'] > df['EMA9']) & (df['MACD'] > df['Signal'])
            df['Sell_Signal'] = (df['Close'] > df['VWAP']) & (df['Close'] > df['EMA9']) & (df['MACD'] > df['Signal'])

            

            latest_buy = df['Buy_Signal'].iloc[-1]
            latest_sell = df['Sell_Signal'].iloc[-1]

            

            if latest_buy:
                buy_signals.append(symbol)
            elif latest_sell:
                sell_signals.append(symbol)
            success = True
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt == 2:
                logging.error(f"Failed to process {symbol} after 3 attempts.")
                logging.error(f"Unhandled error processing {symbol}: {e}")

# Save to file
current_date = datetime.now().strftime("%Y-%m-%d")
filename = f"macd_vwap_signals_{current_date}.txt"

# Save to file
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"MACD Signal Report - {current_date}\n")
    f.write("=" * 40 + "\n\n")

    f.write("Buy Signals :\n")
    f.write("-" * 40 + "\n")
    f.write("\n".join(buy_signals) + "\n\n")

    f.write("Sell Signals :\n")
    f.write("-" * 40 + "\n")
    f.write("\n".join(sell_signals) + "\n")

print(f"Signals saved to {filename}")


