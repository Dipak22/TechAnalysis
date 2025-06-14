import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from datetime import datetime, timedelta

def calculate_signals(ticker, lookback_days=14):
    """Calculate momentum, volume, and moving average signals"""
    try:
        # Download data with volume
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days*3)  # Extra data for indicators
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        #df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < lookback_days:
            return None

        # Calculate indicators
        df['ROC'] = ROCIndicator(close=df['Close'], window=lookback_days).roc()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD_diff'] = macd.macd_diff()
        
        # Moving Averages
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Volume Analysis
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            window=lookback_days
        ).volume_weighted_average_price()
        
        df['OBV'] = OnBalanceVolumeIndicator(
            close=df['Close'],
            volume=df['Volume']
        ).on_balance_volume()
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Price Change
        price_change_pct = (latest['Close'] - df['Close'].iloc[-lookback_days]) / df['Close'].iloc[-lookback_days] * 100
        
        # Volume Signals
        volume_spike = latest['Volume'] > (df['Volume'].rolling(20).mean().iloc[-1] * 2)
        obv_trend = '↑' if latest['OBV'] > prev['OBV'] else '↓'
        
        # Moving Average Signals
        sma_cross = 'Golden Cross' if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50'] else (
                    'Death Cross' if latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50'] else None)
        
        # Generate Trading Signal
        signal = "HOLD"
        if latest['RSI'] > 70 and price_change_pct > 15:
            signal = "SELL (Overbought)"
        elif latest['RSI'] < 30 and price_change_pct < -10:
            signal = "BUY (Oversold)"
        elif (latest['Close'] > latest['SMA_20'] > latest['SMA_50'] and 
              latest['MACD_diff'] > 0 and 
              volume_spike):
            signal = "BUY (Strong Trend)"
        elif (latest['Close'] < latest['SMA_20'] < latest['SMA_50'] and 
              latest['MACD_diff'] < 0):
            signal = "SELL (Downtrend)"
            
        return {
            'Ticker': ticker,
            'Price': f"${latest['Close']:.2f}",
            f'{lookback_days}D Change': f"{price_change_pct:.2f}%",
            'RSI': f"{latest['RSI']:.1f}",
            'SMA_20/50': f"{latest['SMA_20']:.1f}/{latest['SMA_50']:.1f}",
            'SMA Cross': sma_cross if sma_cross else '-',
            'Volume': f"{latest['Volume']/1e6:.1f}M",
            'Volume Spike': 'Yes' if volume_spike else 'No',
            'OBV Trend': obv_trend,
            'Signal': signal,
            'Momentum Score': min(100, max(0, (
                0.3 * price_change_pct + 
                0.2 * latest['RSI'] + 
                20 * (latest['MACD_diff'] > 0) +
                15 * (latest['Close'] > latest['SMA_20'] > latest['SMA_50']) +
                15 * volume_spike
            )))
        }
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

def generate_html_report(results, output_file='momentum_report.html'):
    """Generate enhanced HTML report with all indicators"""
    html_template = f"""
    <html>
    <head>
        <title>Advanced Stock Momentum Report</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            h1 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .buy {{ background-color: #e6ffe6; }}
            .sell {{ background-color: #ffe6e6; }}
            .hold {{ background-color: #ffffe6; }}
            .arrow-up {{ color: green; font-weight: bold; }}
            .arrow-down {{ color: red; font-weight: bold; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .neutral {{ color: gray; }}
            .score-bar {{ 
                height: 20px; 
                border-radius: 3px; 
                background: linear-gradient(90deg, #ff0000 0%, #ffff00 50%, #00ff00 100%);
            }}
            .score-value {{
                display: inline-block;
                width: {100/len(results)}%;
                height: 100%;
                background-color: rgba(255,255,255,0.7);
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Advanced Stock Momentum Report ({datetime.today().strftime('%Y-%m-%d')})</h1>
        <table>
            <tr>
                <th>Rank</th>
                <th>Ticker</th>
                <th>Price</th>
                <th>14D Change</th>
                <th>RSI</th>
                <th>SMA (20/50)</th>
                <th>SMA Cross</th>
                <th>Volume</th>
                <th>Volume Spike</th>
                <th>OBV Trend</th>
                <th>Signal</th>
                <th>Momentum Score</th>
            </tr>
            {"".join([
                f"""
                <tr class="{'buy' if 'BUY' in r['Signal'] else 'sell' if 'SELL' in r['Signal'] else 'hold'}">
                    <td>{i+1}</td>
                    <td><b>{r['Ticker']}</b></td>
                    <td>{r['Price']}</td>
                    <td class="{'positive' if float(r['14D Change'].strip('%')) > 0 else 'negative'}">
                        {r['14D Change']} {'↑' if float(r['14D Change'].strip('%')) > 0 else '↓'}
                    </td>
                    <td class="{('positive' if float(r['RSI']) < 30 else 'negative' if float(r['RSI']) > 70 else 'neutral')}">
                        {r['RSI']}
                    </td>
                    <td>{r['SMA_20/50']}</td>
                    <td class="{('positive' if 'Golden' in r['SMA Cross'] else 'negative' if 'Death' in r['SMA Cross'] else 'neutral')}">
                        {r['SMA Cross']}
                    </td>
                    <td>{r['Volume']}</td>
                    <td class="{('positive' if r['Volume Spike'] == 'Yes' else 'neutral')}">
                        {r['Volume Spike']}
                    </td>
                    <td class="{('positive' if r['OBV Trend'] == '↑' else 'negative')}">
                        {r['OBV Trend']}
                    </td>
                    <td><b class="{('positive' if 'BUY' in r['Signal'] else 'negative' if 'SELL' in r['Signal'] else 'neutral')}">
                        {r['Signal']}
                    </b></td>
                    <td>
                        <div class="score-bar">
                            <div class="score-value" style="margin-left: {r['Momentum Score']}%">
                                {r['Momentum Score']:.0f}
                            </div>
                        </div>
                    </td>
                </tr>
                """ for i, r in enumerate(results)
            ])}
        </table>
        <h3>Indicator Legend:</h3>
        <ul>
            <li><b>SMA Cross</b>: Golden Cross (20MA > 50MA) = Bullish, Death Cross (20MA < 50MA) = Bearish</li>
            <li><b>Volume Spike</b>: Volume > 2x 20-day average</li>
            <li><b>OBV Trend</b>: ↑ = Accumulation, ↓ = Distribution</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:  # Added encoding='utf-8' here
        f.write(html_template)
    print(f"Report generated: {output_file}")

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
    generate_html_report(results)

# Example usage
if __name__ == "__main__":
    stocks = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'GOOGL', 'AMD', 'INTC', 'SPY']
    analyze_stocks(stocks, lookback_days=14)