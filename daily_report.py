import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator
from datetime import datetime, timedelta
from sector_mapping import sector_stocks  # Replace with your stock list
from my_stocks import my_stocks

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Custom PSAR to avoid warnings
class FixedPSARIndicator(PSARIndicator):
    def __init__(self, high, low, close, step=0.02, max_step=0.2):
        super().__init__(high=high, low=low, close=close, step=step, max_step=max_step)
        
    def _calc_psar(self):
        psar = self._close.copy()
        psar.iloc[:] = np.nan
        
        for i in range(2, len(self._close)):
            if i == 2:
                if self._close.iloc[i] > self._close.iloc[i-1]:
                    trend = 1
                    psar.iloc[i] = min(self._low.iloc[i-1], self._low.iloc[i-2])
                else:
                    trend = -1
                    psar.iloc[i] = max(self._high.iloc[i-1], self._high.iloc[i-2])
                continue
                
            # Rest of original logic with .iloc
            prev_psar = psar.iloc[i-1]
            if trend == 1:
                psar.iloc[i] = prev_psar + self._step * (self._high.iloc[i-1] - prev_psar)
                psar.iloc[i] = min(psar.iloc[i], self._low.iloc[i-1], self._low.iloc[i-2])
            else:
                psar.iloc[i] = prev_psar + self._step * (self._low.iloc[i-1] - prev_psar)
                psar.iloc[i] = max(psar.iloc[i], self._high.iloc[i-1], self._high.iloc[i-2])
            
        return psar

def calculate_signals(ticker, short_period=14, medium_period=26, long_period=50):
    """Calculate momentum, volume, and trend signals across three timeframes with PSAR"""
    try:
        # Download data - extended for long-term indicators
        end_date = datetime.today()
        start_date = end_date - timedelta(days=long_period*3)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty or len(df) < long_period:
            return None

        # Initialize indicators dictionary
        indicators = {}
        
        # Short-term indicators (14 days)
        indicators['short'] = {
            'RSI': RSIIndicator(close=df['Close'], window=short_period).rsi(),
            'ROC': ROCIndicator(close=df['Close'], window=short_period).roc(),
            'Stoch_%K': StochasticOscillator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=short_period,
                smooth_window=3
            ).stoch(),
            'SMA': SMAIndicator(close=df['Close'], window=short_period).sma_indicator(),
            'EMA': EMAIndicator(close=df['Close'], window=short_period).ema_indicator(),
            'ATR': AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=short_period
            ).average_true_range(),
            'PSAR': FixedPSARIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                step=0.02,  # Standard acceleration
                max_step=0.2
            ).psar()
        }
        
        # Medium-term indicators (26 days)
        indicators['medium'] = {
            'RSI': RSIIndicator(close=df['Close'], window=medium_period).rsi(),
            'ROC': ROCIndicator(close=df['Close'], window=medium_period).roc(),
            'SMA': SMAIndicator(close=df['Close'], window=medium_period).sma_indicator(),
            'EMA': EMAIndicator(close=df['Close'], window=medium_period).ema_indicator(),
            'MACD': MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).macd(),
            'MACD_diff': MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_diff(),
            'ADX': ADXIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=medium_period
            ).adx(),
            'PSAR': FixedPSARIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                step=0.015,  # Slightly slower
                max_step=0.15
            ).psar()
        }
        
        # Long-term indicators (50 days)
        indicators['long'] = {
            'SMA': SMAIndicator(close=df['Close'], window=long_period).sma_indicator(),
            'EMA': EMAIndicator(close=df['Close'], window=long_period).ema_indicator(),
            'BB': BollingerBands(close=df['Close'], window=long_period, window_dev=2),
            'VWAP': VolumeWeightedAveragePrice(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume'],
                window=long_period
            ).volume_weighted_average_price(),
            'PSAR': FixedPSARIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                step=0.01,  # Slow acceleration for long-term
                max_step=0.1
            ).psar()
        }
        
        # Volume indicators
        volume_indicators = {
            'OBV': OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume(),
            'ADI': AccDistIndexIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume']
            ).acc_dist_index(),
            'Volume_MA': df['Volume'].rolling(20).mean(),
            'Volume_Spike': df['Volume'] > (df['Volume'].rolling(20).mean() * 2)
        }
        
        # Get latest values
        latest = {
            'price': df['Close'].iloc[-1],
            'short': {k: v.iloc[-1] for k, v in indicators['short'].items()},
            'medium': {k: v.iloc[-1] if not isinstance(v, dict) else {sk: sv.iloc[-1] for sk, sv in v.items()} 
                      for k, v in indicators['medium'].items()},
            'long': {k: v.iloc[-1] if not hasattr(v, 'bollinger_hband') else {
                'upper': v.bollinger_hband().iloc[-1],
                'middle': v.bollinger_mavg().iloc[-1],
                'lower': v.bollinger_lband().iloc[-1],
                'percent': (df['Close'].iloc[-1] - v.bollinger_lband().iloc[-1]) / 
                          (v.bollinger_hband().iloc[-1] - v.bollinger_lband().iloc[-1])
            } for k, v in indicators['long'].items()},
            'volume': {k: v.iloc[-1] for k, v in volume_indicators.items()}
        }
        
        # Previous values for trend analysis
        prev = {
            'short': {k: v.iloc[-2] for k, v in indicators['short'].items()},
            'medium': {k: v.iloc[-2] if not isinstance(v, dict) else {sk: sv.iloc[-2] for sk, sv in v.items()} 
                     for k, v in indicators['medium'].items()},
        }
        
        # Price changes
        price_changes = {
            'short': (latest['price'] - df['Close'].iloc[-short_period]) / df['Close'].iloc[-short_period] * 100,
            'medium': (latest['price'] - df['Close'].iloc[-medium_period]) / df['Close'].iloc[-medium_period] * 100,
            'long': (latest['price'] - df['Close'].iloc[-long_period]) / df['Close'].iloc[-long_period] * 100
        }
        
        # Trend analysis with PSAR
        trends = {
            # Moving average trends
            'short_term_up': latest['price'] > latest['short']['SMA'] > latest['short']['EMA'],
            'medium_term_up': latest['price'] > latest['medium']['SMA'] > latest['medium']['EMA'],
            'long_term_up': latest['price'] > latest['long']['SMA'] > latest['long']['EMA'],
            'golden_cross': latest['short']['SMA'] > latest['medium']['SMA'] and prev['short']['SMA'] <= prev['medium']['SMA'],
            'death_cross': latest['short']['SMA'] < latest['medium']['SMA'] and prev['short']['SMA'] >= prev['medium']['SMA'],
            'macd_bullish': latest['medium']['MACD_diff'] > 0 and prev['medium']['MACD_diff'] <= 0,
            'macd_bearish': latest['medium']['MACD_diff'] < 0 and prev['medium']['MACD_diff'] >= 0,
            'adx_strength': latest['medium']['ADX'] > 25,
            
            # PSAR trends
            'sar_short_bullish': latest['price'] > latest['short']['PSAR'],
            'sar_medium_bullish': latest['price'] > latest['medium']['PSAR'],
            'sar_long_bullish': latest['price'] > latest['long']['PSAR'],
            'sar_short_bearish': latest['price'] < latest['short']['PSAR'],
            'sar_medium_bearish': latest['price'] < latest['medium']['PSAR'],
            'sar_long_bearish': latest['price'] < latest['long']['PSAR'],
        }
        
        # Momentum signals
        momentum = {
            'rsi_short': latest['short']['RSI'],
            'rsi_medium': latest['medium']['RSI'],
            'stoch_overbought': latest['short']['Stoch_%K'] > 80,
            'stoch_oversold': latest['short']['Stoch_%K'] < 20,
            'roc_short': latest['short']['ROC'],
            'roc_medium': latest['medium']['ROC'],
            'bb_position': latest['long']['BB']['percent'],
            'atr': latest['short']['ATR']
        }
        
        # Volume signals
        volume = {
            'obv_trend': '↑' if latest['volume']['OBV'] > volume_indicators['OBV'].iloc[-2] else '↓',
            'adi_trend': '↑' if latest['volume']['ADI'] > volume_indicators['ADI'].iloc[-2] else '↓',
            'volume_spike': latest['volume']['Volume_Spike'],
            'vwap_relation': 'above' if latest['price'] > latest['long']['VWAP'] else 'below'
        }
        
        # Generate signal reasons
        signal_reasons = []
        
        # Trend reasons
        if trends['golden_cross']:
            signal_reasons.append("Golden Cross (Short > Medium MA)")
        if trends['death_cross']:
            signal_reasons.append("Death Cross (Short < Medium MA)")
        if trends['sar_short_bullish'] and trends['sar_medium_bullish'] and trends['sar_long_bullish']:
            signal_reasons.append("PSAR Bullish (All Timeframes)")
        elif trends['sar_short_bullish'] or trends['sar_medium_bullish']:
            signal_reasons.append("PSAR Bullish (Partial)")
        if trends['adx_strength']:
            signal_reasons.append("Strong Trend (ADX > 25)")
            
        # Momentum reasons
        if momentum['rsi_short'] > 70:
            signal_reasons.append(f"Short-term RSI Overbought ({momentum['rsi_short']:.1f})")
        if momentum['rsi_short'] < 30:
            signal_reasons.append(f"Short-term RSI Oversold ({momentum['rsi_short']:.1f})")
        if momentum['stoch_overbought']:
            signal_reasons.append("Stochastic Overbought")
        if momentum['stoch_oversold']:
            signal_reasons.append("Stochastic Oversold")
        if momentum['bb_position'] > 0.8:
            signal_reasons.append(f"BB Upper Band ({momentum['bb_position']:.2%})")
        if momentum['bb_position'] < 0.2:
            signal_reasons.append(f"BB Lower Band ({momentum['bb_position']:.2%})")
            
        # Volume reasons
        if volume['volume_spike']:
            signal_reasons.append("Volume Spike (2x MA)")
        if volume['obv_trend'] == '↑' and volume['volume_spike']:
            signal_reasons.append("OBV Uptick with Volume")
        if volume['vwap_relation'] == 'above':
            signal_reasons.append("Price Above VWAP")
            
        # Price change reasons
        if price_changes['short'] > 10:
            signal_reasons.append(f"Short-term Up {price_changes['short']:.1f}%")
        if price_changes['short'] < -5:
            signal_reasons.append(f"Short-term Down {abs(price_changes['short']):.1f}%")
        
        # Generate composite score (0-100)
        score = 50  # Neutral starting point
        
        # Trend factors (30%)
        score += 5 if trends['short_term_up'] else -10
        score += 5 if trends['medium_term_up'] else -5
        score += 5 if trends['long_term_up'] else -3
        score += 5 if trends['golden_cross'] else (-10 if trends['death_cross'] else 0)
        score += 5 if trends['macd_bullish'] else (-5 if trends['macd_bearish'] else 0)
        score += 5 if trends['adx_strength'] else 0
        
        # PSAR factors (15%)
        score += 5 if trends['sar_short_bullish'] else -5
        score += 5 if trends['sar_medium_bullish'] else -5
        score += 5 if trends['sar_long_bullish'] else -5
        
        # Momentum factors (25%)
        score += 5 * ((momentum['rsi_short'] - 50) / 10)  # Normalized RSI contribution
        score += 5 if momentum['stoch_oversold'] else (-5 if momentum['stoch_overbought'] else 0)
        score += 5 * (momentum['roc_short'] / 5)  # Normalized ROC contribution
        score += 5 * (momentum['roc_medium'] / 3)  # Normalized ROC contribution
        score += 5 * (momentum['bb_position'] - 0.5)  # BB position contribution
        
        # Volume factors (20%)
        score += 10 if volume['obv_trend'] == '↑' else -10
        score += 5 if volume['adi_trend'] == '↑' else -3
        score += 5 if volume['volume_spike'] and volume['obv_trend'] == '↑' else (
                 -5 if volume['volume_spike'] and volume['obv_trend'] == '↓' else 0)
        score += 5 if volume['vwap_relation'] == 'above' else -3
        
        # Price action factors (10%)
        score += 5 * (price_changes['short'] / 5)  # Normalized short-term change
        score += 3 * (price_changes['medium'] / 3)  # Normalized medium-term change
        score += 2 * (price_changes['long'] / 2)  # Normalized long-term change
        
        # Cap score between 0 and 100
        score = max(0, min(100, score))
        
        # Generate trading signal
        signal = "HOLD"
        signal_strength = ""
        
        # STRONG BUY: All PSAR timeframes bullish + high score
        if (score > 80 and 
            trends['sar_short_bullish'] and 
            trends['sar_medium_bullish'] and 
            trends['sar_long_bullish'] and 
            volume['obv_trend'] == '↑'):
            signal = "STRONG BUY"
            signal_strength = "(Multi-Timeframe PSAR Confirmed)"
        
        # BUY: Partial PSAR confirmation
        elif (score > 65 and 
              (trends['sar_short_bullish'] or trends['sar_medium_bullish']) and 
              volume['obv_trend'] == '↑'):
            signal = "BUY"
            signal_strength = "(PSAR Bullish)"
            
        # STRONG SELL: All PSAR timeframes bearish + low score
        elif (score < 20 and 
              trends['sar_short_bearish'] and 
              trends['sar_medium_bearish'] and 
              trends['sar_long_bearish'] and 
              volume['obv_trend'] == '↓'):
            signal = "STRONG SELL"
            signal_strength = "(Multi-Timeframe PSAR Confirmed)"
            
        # SELL: Partial PSAR confirmation
        elif (score < 35 and 
              (trends['sar_short_bearish'] or trends['sar_medium_bearish']) and 
              volume['obv_trend'] == '↓'):
            signal = "SELL"
            signal_strength = "(PSAR Bearish)"
        
        # Add reasons if not already included
        if signal != "HOLD" and signal_reasons:
            signal += f" {signal_strength} [{', '.join(signal_reasons)}]"
        elif signal == "HOLD" and signal_reasons:
            signal = f"HOLD (Conflicting signals: {', '.join(signal_reasons)})"
            
        # Prepare result dictionary
        result = {
            'Ticker': ticker,
            'Price': f"{latest['price']:.2f}",
            f'Change_{short_period}D': f"{price_changes['short']:.1f}%",
            f'Change_{medium_period}D': f"{price_changes['medium']:.1f}%",
            f'Change_{long_period}D': f"{price_changes['long']:.1f}%",
            f'RSI_{short_period}': f"{momentum['rsi_short']:.1f}",
            f'RSI_{medium_period}': f"{momentum['rsi_medium']:.1f}",
            f'Stoch_%K_{short_period}': f"{latest['short']['Stoch_%K']:.1f}",
            f'MACD_diff_{medium_period}': f"{latest['medium']['MACD_diff']:.3f}",
            'BB_%': f"{momentum['bb_position']:.2%}",
            f'SMA_{short_period}/{medium_period}/{long_period}': f"{latest['short']['SMA']:.1f}/{latest['medium']['SMA']:.1f}/{latest['long']['SMA']:.1f}",
            f'PSAR_{short_period}': 'Bullish' if trends['sar_short_bullish'] else 'Bearish',
            f'PSAR_{medium_period}': 'Bullish' if trends['sar_medium_bullish'] else 'Bearish',
            f'PSAR_{long_period}': 'Bullish' if trends['sar_long_bullish'] else 'Bearish',
            'Volume': f"{df['Volume'].iloc[-1]/1e6:.1f}M",
            'Volume_Spike': 'Yes' if volume['volume_spike'] else 'No',
            'OBV_Trend': volume['obv_trend'],
            'VWAP_Relation': volume['vwap_relation'],
            'Score': f"{score:.1f}",
            'Signal': signal
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

def generate_html_report(short_period,medium_period, long_period, results, output_file='momentum_report.html'):
    """Generate HTML report with dynamic period headers"""
    
    # Extract periods from the first result's keys
    def extract_period(key_prefix):
        for key in results[0].keys():
            if key.startswith(key_prefix):
                return key.split('_')[-1].replace('D','')
        return None
    
    short_period =str(short_period)
    medium_period = str(medium_period)  # Second RSI is medium period
    long_period = str(long_period)   # Last PSAR is long period

    html_template = f"""
    <html>
    <head>
        <title>Multi-Timeframe Stock Analysis Report</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            h1 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .buy {{ background-color: #e6ffe6; }}
            .strong-buy {{ background-color: #ccffcc; }}
            .sell {{ background-color: #ffe6e6; }}
            .strong-sell {{ background-color: #ffcccc; }}
            .hold {{ background-color: #ffffe6; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .neutral {{ color: gray; }}
            .score-bar {{
                height: 20px;
                background: linear-gradient(to right, red, yellow, green);
                position: relative;
                border-radius: 3px;
            }}
            .score-value {{
                position: absolute;
                left: 0;
                top: 0;
                bottom: 0;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>Multi-Timeframe Stock Analysis Report ({datetime.today().strftime('%Y-%m-%d')})</h1>
        <h3>Analysis Periods: Short ({short_period}D), Medium ({medium_period}D), Long ({long_period}D)</h3>
        <table>
            <tr>
                <th>Rank</th>
                <th>Ticker</th>
                <th>Price</th>
                <th>{short_period}D Chg</th>
                <th>{medium_period}D Chg</th>
                <th>{long_period}D Chg</th>
                <th>RSI ({short_period}/{medium_period})</th>
                <th>Stoch %K</th>
                <th>MACD Diff</th>
                <th>BB %</th>
                <th>SMA ({short_period}/{medium_period}/{long_period})</th>
                <th>PSAR ({short_period}/{medium_period}/{long_period})</th>
                <th>Volume</th>
                <th>Score</th>
                <th>Signal</th>
            </tr>
            {"".join([
                f"""
                <tr class="{{
                    'strong-buy' if 'STRONG BUY' in r['Signal'] else
                    'buy' if 'BUY' in r['Signal'] else
                    'strong-sell' if 'STRONG SELL' in r['Signal'] else
                    'sell' if 'SELL' in r['Signal'] else 'hold'
                }}">
                    <td>{i+1}</td>
                    <td><b>{r['Ticker']}</b></td>
                    <td>{r['Price']}</td>
                    <td class="{{'positive' if float(r[f'Change_{short_period}D'].strip('%')) > 0 else 'negative'}}">
                        {r[f'Change_{short_period}D']}
                    </td>
                    <td class="{{'positive' if float(r[f'Change_{medium_period}D'].strip('%')) > 0 else 'negative'}}">
                        {r[f'Change_{medium_period}D']}
                    </td>
                    <td class="{{'positive' if float(r[f'Change_{long_period}D'].strip('%')) > 0 else 'negative'}}">
                        {r[f'Change_{long_period}D']}
                    </td>
                    <td>
                        <span class="{{'positive' if float(r[f'RSI_{short_period}']) < 30 else 'negative' if float(r[f'RSI_{short_period}']) > 70 else 'neutral'}}">
                            {r[f'RSI_{short_period}']}
                        </span>/
                        <span class="{{'positive' if float(r[f'RSI_{medium_period}']) < 30 else 'negative' if float(r[f'RSI_{medium_period}']) > 70 else 'neutral'}}">
                            {r[f'RSI_{medium_period}']}
                        </span>
                    </td>
                    <td class="{{'positive' if float(r[f'Stoch_%K_{short_period}']) < 20 else 'negative' if float(r[f'Stoch_%K_{short_period}']) > 80 else 'neutral'}}">
                        {r[f'Stoch_%K_{short_period}']}
                    </td>
                    <td class="{{'positive' if float(r[f'MACD_diff_{medium_period}']) > 0 else 'negative'}}">
                        {r[f'MACD_diff_{medium_period}']}
                    </td>
                    <td class="{{'positive' if float(r['BB_%'].strip('%'))/100 < 0.2 else 'negative' if float(r['BB_%'].strip('%'))/100 > 0.8 else 'neutral'}}">
                        {r['BB_%']}
                    </td>
                    <td>{r[f'SMA_{short_period}/{medium_period}/{long_period}']}</td>
                    <td>
                        <span class="{{'positive' if r[f'PSAR_{short_period}'] == 'Bullish' else 'negative'}}">
                            {r[f'PSAR_{short_period}']}
                        </span>/
                        <span class="{{'positive' if r[f'PSAR_{medium_period}'] == 'Bullish' else 'negative'}}">
                            {r[f'PSAR_{medium_period}']}
                        </span>/
                        <span class="{{'positive' if r[f'PSAR_{long_period}'] == 'Bullish' else 'negative'}}">
                            {r[f'PSAR_{long_period}']}
                        </span>
                    </td>
                    <td>
                        {r['Volume']} 
                        <span class="{{'positive' if r['Volume_Spike'] == 'Yes' else 'neutral'}}">
                            ({r['Volume_Spike']}) {r['OBV_Trend']}
                        </span>
                    </td>
                    <td>
                        <div class="score-bar">
                            <div class="score-value">{r['Score']}</div>
                        </div>
                    </td>
                    <td><b class="{{'positive' if 'BUY' in r['Signal'] else 'negative' if 'SELL' in r['Signal'] else 'neutral'}}">
                        {r['Signal']}
                    </b></td>
                </tr>
                """ for i, r in enumerate(results)
            ])}
        </table>
        <h3>Indicator Legend:</h3>
        <ul>
            <li><b>Timeframes</b>: Short ({short_period}D), Medium ({medium_period}D), Long ({long_period}D)</li>
            <li><b>PSAR</b>: Bullish (Price > SAR), Bearish (Price < SAR)</li>
            <li><b>Score</b>: 0-100 (Higher = Stronger Bullish)</li>
            <li><b>Signal Strength</b>:
                <ul>
                    <li><span class="strong-buy">STRONG BUY</span>: All PSAR timeframes bullish + score >80</li>
                    <li><span class="buy">BUY</span>: Partial PSAR bullish + score >65</li>
                    <li><span class="strong-sell">STRONG SELL</span>: All PSAR timeframes bearish + score <20</li>
                    <li><span class="sell">SELL</span>: Partial PSAR bearish + score <35</li>
                </ul>
            </li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Report generated: {output_file}")

def analyze_stocks(stock_list, short_period=14, medium_period=26, long_period=50):
    """Main analysis function"""
    results = []
    for ticker in stock_list:
        print(f"Processing {ticker}...")
        data = calculate_signals(ticker, short_period, medium_period, long_period)
        if data:
            results.append(data)
    
    if not results:
        print("No valid results.")
        return
    
    # Sort by score
    results.sort(key=lambda x: float(x['Score']), reverse=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_FILE = f"momentum_report_{current_date}.html"
    generate_html_report(short_period,medium_period, long_period, results, output_file=OUTPUT_FILE)

# Example usage
if __name__ == "__main__":
    stocks = my_stocks # Replace with your stock list
    analyze_stocks(stocks, short_period=10, medium_period=20, long_period=50)