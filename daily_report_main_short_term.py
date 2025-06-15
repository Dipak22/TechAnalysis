import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from datetime import datetime, timedelta
from my_stocks import my_stocks
from sector_mapping import sector_stocks

def calculate_signals(ticker, lookback_days=7):
    """Optimized for 5-7 day momentum"""
    try:
        # Download data (extra buffer for indicators)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days*2)  
        stock = yf.Ticker(ticker)
        df =  stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty or len(df) < lookback_days:
            return None

        # --- Momentum Indicators (Shorter Windows) ---
        df['ROC'] = ROCIndicator(close=df['Close'], window=lookback_days).roc()
        df['RSI'] = RSIIndicator(close=df['Close'], window=10).rsi()  # Shorter RSI
        macd = MACD(close=df['Close'], window_slow=12, window_fast=5)  # Faster MACD
        df['MACD_diff'] = macd.macd_diff()
        
        # --- Moving Averages (Faster) ---
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()  # Fast EMA
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()  # Medium EMA
        
        # --- Volume Analysis (More Sensitive) ---
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['High'], low=df['Low'], close=df['Close'], 
            volume=df['Volume'], window=lookback_days
        ).volume_weighted_average_price()
        
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        
        # --- Latest Values ---
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Price Change (Short-Term)
        price_change_pct = (latest['Close'] - df['Close'].iloc[-lookback_days]) / df['Close'].iloc[-lookback_days] * 100
        
        # Volume Spike (Tighter Threshold)
        volume_spike = latest['Volume'] > (df['Volume'].rolling(5).mean().iloc[-1] * 1.8)  # 1.8x vs 2x
        
        # EMA Cross (Faster)
        ema_cross = 'Bullish' if latest['EMA_5'] > latest['EMA_10'] and prev['EMA_5'] <= prev['EMA_10'] else (
                   'Bearish' if latest['EMA_5'] < latest['EMA_10'] and prev['EMA_5'] >= prev['EMA_10'] else None)
        
        # --- Generate Signal (Optimized for Short-Term) ---
        signal = "HOLD"
        if latest['RSI'] > 65 and price_change_pct > 8:  # Lower thresholds
            signal = "SELL (Overbought)"
        elif latest['RSI'] < 35 and price_change_pct < -5:
            signal = "BUY (Oversold)"
        elif (latest['Close'] > latest['EMA_5'] > latest['EMA_10'] and 
              latest['MACD_diff'] > 0 and 
              volume_spike):
            signal = "BUY (Strong Uptrend)"
        elif (latest['Close'] < latest['EMA_5'] < latest['EMA_10'] and 
              latest['MACD_diff'] < 0):
            signal = "SELL (Downtrend)"
            
        return {
            'Ticker': ticker,
            'Price': f"{latest['Close']:.2f}",
            f'{lookback_days}D Change': f"{price_change_pct:.2f}%",
            'RSI': f"{latest['RSI']:.1f}",
            'EMA_5/10': f"{latest['EMA_5']:.1f}/{latest['EMA_10']:.1f}",
            'EMA Cross': ema_cross if ema_cross else '-',
            'Volume Spike': 'Yes' if volume_spike else 'No',
            'MACD_diff': latest['MACD_diff'],
            'Signal': signal,
            'Momentum Score': min(100, max(0, (
                0.4 * price_change_pct +  # Weight recent price change more
                0.2 * latest['RSI'] + 
                20 * (latest['MACD_diff'] > 0) +
                20 * (latest['Close'] > latest['EMA_5'] > latest['EMA_10']) +
                20 * volume_spike
            ))),
            'Period': lookback_days 
        }
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

def generate_html_report(results, output_file='momentum_report.html'):
    """Generate HTML report optimized for short-term trading signals"""
    html_template = f"""
    <html>
    <head>
        <title>Short-Term Momentum Report ({datetime.today().strftime('%Y-%m-%d')})</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            h1 {{ color: #333366; }}
            .subtitle {{ color: #666; font-style: italic; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            th, td {{ padding: 8px; text-align: center; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .buy {{ background-color: #e6f7e6; }}
            .sell {{ background-color: #ffe6e6; }}
            .hold {{ background-color: #ffffcc; }}
            .positive {{ color: #008000; font-weight: bold; }}
            .negative {{ color: #ff0000; font-weight: bold; }}
            .neutral {{ color: #666; }}
            .arrow-up {{ content: "↑"; color: green; }}
            .arrow-down {{ content: "↓"; color: red; }}
            .score-cell {{
                background: linear-gradient(90deg, #ff0000 0%, #ffff00 50%, #00ff00 100%);
                background-size: 100% 100%;
                position: relative;
            }}
            .score-value {{
                position: absolute;
                top: 0;
                left: 0;
                width: {100/len(results)}%;
                height: 100%;
                background-color: rgba(255,255,255,0.7);
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Short-Term Momentum Scanner</h1>
        <p class="subtitle">Generated on {datetime.today().strftime('%Y-%m-%d %H:%M')} | Period: {results[0]['Period']} days</p>
        
        <table>
            <tr>
                <th>Rank</th>
                <th>Ticker</th>
                <th>Price</th>
                <th>Change</th>
                <th>RSI (10)</th>
                <th>EMA (5/10)</th>
                <th>EMA Cross</th>
                <th>Volume Spike</th>
                <th>MACD</th>
                <th>Signal</th>
                <th>Momentum</th>
            </tr>
            {"".join([
    f"""
    <tr class="{'buy' if 'BUY' in r['Signal'] else 'sell' if 'SELL' in r['Signal'] else 'hold'}">
        <td>{i+1}</td>
        <td><b>{r['Ticker']}</b></td>
        <td>{r['Price']}</td>
        <td class="{'positive' if float(r[f"{r['Period']}D Change"].strip('%')) > 0 else 'negative'}">
            {r[f"{r['Period']}D Change"]} {'↑' if float(r[f"{r['Period']}D Change"].strip('%')) > 0 else '↓'}
        </td>
        <td class="{('positive' if float(r['RSI']) < 35 else 'negative' if float(r['RSI']) > 65 else 'neutral')}">
            {r['RSI']}
        </td>
        <td>{r['EMA_5/10']}</td>
        <td class="{('positive' if 'Bullish' in r['EMA Cross'] else 'negative' if 'Bearish' in r['EMA Cross'] else 'neutral')}">
            {r['EMA Cross']}
        </td>
        <td class="{('positive' if r['Volume Spike'] == 'Yes' else 'neutral')}">
            {r['Volume Spike']}
        </td>
        <td class="{('positive' if float(r['MACD_diff']) > 0 else 'negative')}">
            {'↑' if float(r['MACD_diff']) > 0 else '↓'}
        </td>
        <td><b class="{('positive' if 'BUY' in r['Signal'] else 'negative' if 'SELL' in r['Signal'] else 'neutral')}">
            {r['Signal']}
        </b></td>
        <td class="score-cell">
            <div class="score-value">
                {r['Momentum Score']:.0f}
            </div>
        </td>
    </tr>
    """ for i, r in enumerate(results)
])}
        </table>

        <h3>Key Metrics Legend:</h3>
        <ul>
            <li><b>EMA Cross</b>: Bullish (5EMA > 10EMA) | Bearish (5EMA < 10EMA)</li>
            <li><b>Volume Spike</b>: Volume > 1.8x 5-day average</li>
            <li><b>RSI (10)</b>: Oversold (<35) | Overbought (>65)</li>
            <li><b>MACD</b>: ↑ = Bullish momentum | ↓ = Bearish momentum</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Short-term report generated: {output_file}")

def analyze_stocks(stock_list, lookback_days=14):
    """Main analysis function"""
    results = []
    for ticker in stock_list:
        print(f"Processing {ticker}...")
        data = calculate_signals(ticker, lookback_days)
        if data:
            data['Period'] = lookback_days
            results.append(data)
    
    if not results:
        print("No valid results.")
        return
    
    # Sort by momentum score
    results.sort(key=lambda x: x['Momentum Score'], reverse=True)
    # Output file name
    current_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_FILE = f"momentum_report_short_term_{current_date}.html"
    generate_html_report(results, output_file=OUTPUT_FILE)

# Example usage
if __name__ == "__main__":
    stocks = [stock for stocks in sector_stocks.values() for stock in stocks]  # Replace with your stock list
    analyze_stocks(stocks, lookback_days=10)