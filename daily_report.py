import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from datetime import datetime, timedelta
from sector_mapping import sector_stocks  # Assuming this is a list of stock tickers

def calculate_signals(ticker, lookback_days=14):
    """Calculate momentum, volume, and moving average signals"""
    try:
        # Download data with volume
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days*3)  # Extra data for indicators
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty or len(df) < lookback_days:
            return None

        # Calculate indicators
        df['ROC'] = ROCIndicator(close=df['Close'], window=lookback_days).roc()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD_diff'] = macd.macd_diff()
        
        # Moving Averages - Changed to 14 and 26 days
        df['SMA_14'] = SMAIndicator(close=df['Close'], window=14).sma_indicator()
        df['SMA_26'] = SMAIndicator(close=df['Close'], window=26).sma_indicator()
        
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
        df['SMA_26_rising'] = df['SMA_26'] > df['SMA_26'].shift(5)
        
        # Price Change
        price_change_pct = (latest['Close'] - df['Close'].iloc[-lookback_days]) / df['Close'].iloc[-lookback_days] * 100
        
        # Volume Signals
        volume_spike = latest['Volume'] > (df['Volume'].rolling(20).mean().iloc[-1] * 2)
        obv_trend = '↑' if latest['OBV'] > prev['OBV'] else '↓'
        
        # Moving Average Signals
        sma_cross = 'Golden Cross' if latest['SMA_14'] > latest['SMA_26'] and prev['SMA_14'] <= prev['SMA_26'] else (
                    'Death Cross' if latest['SMA_14'] < latest['SMA_26'] and prev['SMA_14'] >= prev['SMA_26'] else None)
        
        # Generate Trading Signal
        signal = "HOLD"
        signal_reasons = []

        # Overbought/Oversold Conditions
        if latest['RSI'] > 70:
            signal_reasons.append("RSI Overbought (>70)")
        if latest['RSI'] < 30:
            signal_reasons.append("RSI Oversold (<30)")
        if price_change_pct > 15:
            signal_reasons.append(f"Large Price Gain ({price_change_pct:.1f}%)")
        if price_change_pct < -10:
            signal_reasons.append(f"Large Price Drop ({price_change_pct:.1f}%)")

        # Trend Conditions
        sma_bullish = latest['Close'] > latest['SMA_14'] > latest['SMA_26']
        sma_bearish = latest['Close'] < latest['SMA_14'] < latest['SMA_26']
        macd_bullish = latest['MACD_diff'] > 0
        macd_bearish = latest['MACD_diff'] < 0

        # Volume Conditions
        vwap_bullish = latest['Close'] > latest['VWAP']
        vwap_bearish = latest['Close'] < latest['VWAP']
        obv_bullish = obv_trend == '↑'
        obv_bearish = obv_trend == '↓'

        # Momentum Conditions
        roc_bullish = latest['ROC'] > 0
        roc_bearish = latest['ROC'] < 0

        # Composite Signal Logic
        bullish_count = 0
        bearish_count = 0

        # Count bullish indicators
        if sma_bullish: bullish_count += 1
        if macd_bullish: bullish_count += 1
        if vwap_bullish: bullish_count += 1
        if obv_bullish: bullish_count += 1
        if roc_bullish: bullish_count += 1
        if volume_spike and obv_bullish: bullish_count += 1  # Extra weight for volume confirmation

        # Count bearish indicators
        if sma_bearish: bearish_count += 1
        if macd_bearish: bearish_count += 1
        if vwap_bearish: bearish_count += 1
        if obv_bearish: bearish_count += 1
        if roc_bearish: bearish_count += 1
        if volume_spike and obv_bearish: bearish_count += 1  # Extra weight for volume confirmation

        # Strong Buy Signal (multiple confirmations)
        if (bullish_count >= 4 and 
            latest['RSI'] < 60 and  # Not overbought
            not any("RSI Overbought" in r for r in signal_reasons)):
            signal = "STRONG BUY"
        elif (bullish_count >= 3 and 
            latest['RSI'] < 45 and  # More conservative
            any("RSI Oversold" in r for r in signal_reasons)):
            signal = "BUY"

        # Strong Sell Signal (multiple confirmations)
        elif (bearish_count >= 4 and 
            latest['RSI'] > 40 and  # Not oversold
            not any("RSI Oversold" in r for r in signal_reasons)):
            signal = "STRONG SELL"
        elif (bearish_count >= 3 and 
            latest['RSI'] > 55 and  # More conservative
            any("RSI Overbought" in r for r in signal_reasons)):
            signal = "SELL"

        # Extreme RSI conditions take precedence
        if latest['RSI'] > 70 and price_change_pct > 15:
            signal = "STRONG SELL (Extreme Overbought)"
        elif latest['RSI'] < 30 and price_change_pct < -10:
            signal = "STRONG BUY (Extreme Oversold)"

        # Add reasons to signal if not already included
        if signal != "HOLD" and len(signal_reasons) > 0:
            signal += f" ({', '.join(signal_reasons)})"
        elif signal == "HOLD" and len(signal_reasons) > 0:
            signal = f"HOLD (Conflicting: {', '.join(signal_reasons)})"

            
        return {
            'Ticker': ticker,
            'Price': f"{latest['Close']:.2f}",
            f'{lookback_days}D Change': f"{price_change_pct:.2f}%",
            'RSI': f"{latest['RSI']:.1f}",
            'SMA_14/26': f"{latest['SMA_14']:.1f}/{latest['SMA_26']:.1f}",
            'SMA Cross': sma_cross if sma_cross else '-',
            'Volume': f"{latest['Volume']/1e6:.1f}M",
            'Volume Spike': 'Yes' if volume_spike else 'No',
            'OBV Trend': obv_trend,
            'Signal': signal,
            'Momentum Score': min(100, max(0, (
                # Price Momentum (30%)
                (0.15 * price_change_pct if not pd.isna(price_change_pct) else 0) +
                (0.10 * latest['ROC'] if not pd.isna(latest['ROC']) else 0) +
                (0.05 * (100 if (latest['Close'] > latest['SMA_26'] and df['SMA_26_rising'].iloc[-1]) else 0)) +
                
                # Trend Strength (25%)
                (0.10 * (100 if (latest['Close'] > latest['SMA_14'] > latest['SMA_26']) else 
                        -50 if (latest['Close'] < latest['SMA_14'] < latest['SMA_26']) else 0)) +
                (0.10 * latest['MACD_diff'] * 10 if not pd.isna(latest['MACD_diff']) else 0) +
                (0.05 * (50 if sma_cross == 'Golden Cross' else 
                        -50 if sma_cross == 'Death Cross' else 0)) +
                
                # Volume Confirmation (20%)
                (0.10 * (100 if (volume_spike and obv_trend == '↑') else 
                        -100 if (volume_spike and obv_trend == '↓') else 0)) +
                (0.05 * (50 if latest['Close'] > latest['VWAP'] else -25)) +
                (0.05 * (latest['OBV'] / max(1, df['OBV'].max()) * 50 if not pd.isna(latest['OBV']) else 0)) +
                
                # Momentum Oscillators (15%)
                (0.075 * latest['RSI'] if not pd.isna(latest['RSI']) else 0) +
                (0.075 * (100 if (30 < latest['RSI'] < 70) else 
                        150 if (latest['RSI'] < 30 or latest['RSI'] > 70) else 0)) +
                
                # Risk Adjustment (10%)
                (-0.05 * abs(latest['RSI'] - 50) if not pd.isna(latest['RSI']) else 0) +
                (0.05 * (df['Close'].pct_change().std() * 100 if not pd.isna(df['Close'].pct_change().std()) else 0))
            ) )),
            'Period': lookback_days 
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
                <th>SMA (14/26)</th>
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
                    <td class="{'positive' if float(r[f"{r['Period']}D Change"].strip('%')) > 0 else 'negative'}">
                        {r[f"{r['Period']}D Change"]} {'↑' if float(r[f"{r['Period']}D Change"].strip('%')) > 0 else '↓'}
                    </td>
                    <td class="{('positive' if float(r['RSI']) < 30 else 'negative' if float(r['RSI']) > 70 else 'neutral')}">
                        {r['RSI']}
                    </td>
                    <td>{r['SMA_14/26']}</td>
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
            <li><b>SMA Cross</b>: Golden Cross (14MA > 26MA) = Bullish, Death Cross (14MA < 26MA) = Bearish</li>
            <li><b>Volume Spike</b>: Volume > 2x 20-day average</li>
            <li><b>OBV Trend</b>: ↑ = Accumulation, ↓ = Distribution</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
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
    current_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_FILE = f"momentum_report_{current_date}.html"
    generate_html_report(results, output_file=OUTPUT_FILE)

# Example usage
if __name__ == "__main__":
    stocks = [stock for stocks in sector_stocks.values() for stock in stocks]  # Replace with your stock list
    analyze_stocks(stocks, lookback_days=14)