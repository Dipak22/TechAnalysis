import yfinance as yf
import pandas as pd
import ta
from datetime import datetime
from sector_mapping import sector_stocks
import time
import logging

# Configure logging
logging.basicConfig(filename='macd_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Flatten all stocks from all sectors
#all_stocks = [stock for stocks in sector_stocks.values() for stock in stocks]
all_stocks = [ "NEWGEN.NS","MAPMYINDIA.NS","INTELLECT.NS","INSPIRISYS.NS"]

matching_stocks = []

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

            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['Signal'] = macd.macd_signal()

            if df['MACD'].iloc[-1] >= df['Signal'].iloc[-1]:
                matching_stocks.append(symbol)
            success = True
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt == 2:
                logging.error(f"Failed to process {symbol} after 3 attempts.")
                logging.error(f"Unhandled error processing {symbol}: {e}")

# Save to file
current_date = datetime.now().strftime("%Y-%m-%d")
filename = f"macd_cross_stocks_{current_date}.txt"

with open(filename, "w") as f:
    f.write("Stocks where MACD >= Signal on {}\n".format(current_date))
    f.write("\n".join(matching_stocks))

print(f"Matching stocks saved to {filename}")
