import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator
from datetime import datetime, timedelta
from sector_mapping import sector_stocks  # Replace with your stock list
from my_stocks import my_stocks, PENNY_STOCKS

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

def calculate_signals(ticker, current_date = datetime.today(),short_period=5, medium_period=10, long_period=20):
    """Calculate momentum, volume, and trend signals across three timeframes with PSAR and candlestick patterns"""
    try:
        # Download data - extended for long-term indicators
        end_date = current_date
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
            'open': df['Open'].iloc[-1],
            'high': df['High'].iloc[-1],
            'low': df['Low'].iloc[-1],
            'close': df['Close'].iloc[-1],
            'prev_close': df['Close'].iloc[-2],
            'prev_open': df['Open'].iloc[-2],
            'prev_high': df['High'].iloc[-2],
            'prev_low': df['Low'].iloc[-2],
            'short': {k: v.iloc[-1] for k, v in indicators['short'].items()},
            'medium': {k: v.iloc[-1] if not isinstance(v, dict) else {sk: sv.iloc[-1] for sk, sv in v.items()} 
                      for k, v in indicators['medium'].items()},
            'long': {k: v.iloc[-1] if not hasattr(v, 'bollinger_hband') else {
                'upper': v.bollinger_hband().iloc[-1],
                'middle': v.bollinger_mavg().iloc[-1],
                'lower': v.bollinger_lband().iloc[-1],
                'percent': np.divide(
                                (df['Close'].iloc[-1] - v.bollinger_lband().iloc[-1]),
                                (v.bollinger_hband().iloc[-1] - v.bollinger_lband().iloc[-1]),
                                out=np.zeros(1),
                                where=(v.bollinger_hband().iloc[-1] - v.bollinger_lband().iloc[-1]) != 0
                )[0]
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

        # Detect candlestick patterns (using last 3 candles for better reliability)
        def is_bullish_engulfing():
            return (latest['prev_close'] < latest['prev_open'] and  # Previous candle is bearish
                    latest['close'] > latest['open'] and            # Current candle is bullish
                    latest['close'] > latest['prev_open'] and       # Current close > previous open
                    latest['open'] < latest['prev_close'])          # Current open < previous close

        def is_bearish_engulfing():
            return (latest['prev_close'] > latest['prev_open'] and  # Previous candle is bullish
                    latest['close'] < latest['open'] and            # Current candle is bearish
                    latest['close'] < latest['prev_open'] and       # Current close < previous open
                    latest['open'] > latest['prev_close'])          # Current open > previous close

        def is_hammer():
            body = abs(latest['close'] - latest['open'])
            lower_wick = latest['low'] - min(latest['close'], latest['open'])
            upper_wick = max(latest['close'], latest['open']) - latest['high']
            return (lower_wick >= 2 * body and  # Long lower wick
                    upper_wick <= body * 0.5 and  # Small or no upper wick
                    latest['close'] > latest['open'])  # Bullish candle

        def is_shooting_star():
            body = abs(latest['close'] - latest['open'])
            lower_wick = latest['low'] - min(latest['close'], latest['open'])
            upper_wick = max(latest['close'], latest['open']) - latest['high']
            return (upper_wick >= 2 * body and  # Long upper wick
                    lower_wick <= body * 0.5 and  # Small or no lower wick
                    latest['close'] < latest['open'])  # Bearish candle

        def is_morning_star():
            if len(df) < 3:
                return False
            prev2_close = df['Close'].iloc[-3]
            prev2_open = df['Open'].iloc[-3]
            return (latest['prev_close'] < latest['prev_open'] and  # Middle candle is bearish
                    prev2_close < prev2_open and                    # First candle is bearish
                    latest['close'] > latest['open'] and            # Third candle is bullish
                    latest['close'] > (latest['prev_open'] + latest['prev_close'])/2)  # Closes above midpoint

        def is_evening_star():
            if len(df) < 3:
                return False
            prev2_close = df['Close'].iloc[-3]
            prev2_open = df['Open'].iloc[-3]
            return (latest['prev_close'] > latest['prev_open'] and  # Middle candle is bullish
                    prev2_close > prev2_open and                    # First candle is bullish
                    latest['close'] < latest['open'] and            # Third candle is bearish
                    latest['close'] < (latest['prev_open'] + latest['prev_close'])/2)  # Closes below midpoint

        def is_piercing_line():
            return (latest['prev_close'] < latest['prev_open'] and  # Previous candle is bearish
                    latest['close'] > latest['open'] and            # Current candle is bullish
                    latest['open'] < latest['prev_close'] and       # Opens below previous close
                    latest['close'] > (latest['prev_open'] + latest['prev_close'])/2)  # Closes above midpoint

        def is_dark_cloud_cover():
            return (latest['prev_close'] > latest['prev_open'] and  # Previous candle is bullish
                    latest['close'] < latest['open'] and            # Current candle is bearish
                    latest['open'] > latest['prev_close'] and       # Opens above previous close
                    latest['close'] < (latest['prev_open'] + latest['prev_close'])/2)  # Closes below midpoint

        # Trend analysis with PSAR and candlestick patterns
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
            
            # Candlestick patterns
            'bullish_engulfing': is_bullish_engulfing(),
            'bearish_engulfing': is_bearish_engulfing(),
            'hammer': is_hammer(),
            'shooting_star': is_shooting_star(),
            'morning_star': is_morning_star(),
            'evening_star': is_evening_star(),
            'piercing_line': is_piercing_line(),
            'dark_cloud_cover': is_dark_cloud_cover(),
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

        # Candlestick pattern reasons
        if trends['bullish_engulfing']:
            signal_reasons.append("Bullish Engulfing Pattern")
        if trends['bearish_engulfing']:
            signal_reasons.append("Bearish Engulfing Pattern")
        if trends['hammer']:
            signal_reasons.append("Hammer Pattern (Bullish)")
        if trends['shooting_star']:
            signal_reasons.append("Shooting Star (Bearish)")
        if trends['morning_star']:
            signal_reasons.append("Morning Star Pattern (Strong Bullish)")
        if trends['evening_star']:
            signal_reasons.append("Evening Star Pattern (Strong Bearish)")
        if trends['piercing_line']:
            signal_reasons.append("Piercing Line (Bullish)")
        if trends['dark_cloud_cover']:
            signal_reasons.append("Dark Cloud Cover (Bearish)")

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

        # 1. Trend Strength (25% weight)
        trend_score = 0
        # Weighted by timeframe importance (short < medium < long)
        trend_score += 3 if trends['short_term_up'] else -3
        trend_score += 5 if trends['medium_term_up'] else -5
        trend_score += 7 if trends['long_term_up'] else -7

        # Major trend signals get higher weights
        trend_score += 10 if trends['golden_cross'] else (-10 if trends['death_cross'] else 0)
        trend_score += 7 if trends['macd_bullish'] else (-7 if trends['macd_bearish'] else 0)
        trend_score += 8 if trends['adx_strength'] else -3  # ADX measures trend strength

        # Normalize trend score to 25 points max
        score += (trend_score / 42) * 25  # 42 is max possible trend score

        # 2. PSAR Confirmation (15% weight)
        psar_score = 0
        # Weighted by timeframe importance
        psar_score += 3 if trends['sar_short_bullish'] else -3
        psar_score += 5 if trends['sar_medium_bullish'] else -5
        psar_score += 7 if trends['sar_long_bullish'] else -7

        # Normalize PSAR score to 15 points max
        score += (psar_score / 15) * 15  # 15 is max possible PSAR score

        # 3. Candlestick Patterns (15% weight - high impact but short-term)
        pattern_score = 0
        # Strong reversal patterns
        pattern_score += 15 if trends['morning_star'] else 0
        pattern_score += -15 if trends['evening_star'] else 0
        # Medium strength patterns
        pattern_score += 10 if trends['bullish_engulfing'] else 0
        pattern_score += -10 if trends['bearish_engulfing'] else 0
        pattern_score += 8 if trends['hammer'] else 0
        pattern_score += -8 if trends['shooting_star'] else 0
        pattern_score += 5 if trends['piercing_line'] else 0
        pattern_score += -5 if trends['dark_cloud_cover'] else 0

        # Normalize pattern score to 15 points max (using highest single pattern)
        pattern_score = max(-15, min(15, pattern_score))
        score += pattern_score

        # 4. Momentum Strength (20% weight)
        momentum_score = 0
        # RSI - parabolic weighting (more extreme = stronger signal)
        rsi_weight = 0
        if momentum['rsi_short'] > 70:
            rsi_weight = -((momentum['rsi_short'] - 70) / 30) ** 2  # Overbought penalty
        elif momentum['rsi_short'] < 30:
            rsi_weight = ((30 - momentum['rsi_short']) / 30) ** 2  # Oversold bonus
        momentum_score += rsi_weight * 10

        # Stochastic - similar parabolic weighting
        stoch_weight = 0
        if momentum['stoch_overbought']:
            stoch_weight = -((latest['short']['Stoch_%K'] - 80) / 20) ** 2
        elif momentum['stoch_oversold']:
            stoch_weight = ((20 - latest['short']['Stoch_%K']) / 20) ** 2
        momentum_score += stoch_weight * 5

        # ROC - linear but scaled
        momentum_score += (momentum['roc_short'] / 10) * 3  # Normalized to 3% ROC = 1 point
        momentum_score += (momentum['roc_medium'] / 7) * 2  # Normalized to 3.5% ROC = 1 point

        # Bollinger Bands - position matters
        momentum_score += (momentum['bb_position'] - 0.5) * 10  # 0-1 range becomes -5 to +5

        # Normalize momentum score to 20 points max
        score += (momentum_score / 30) * 20  # 30 is approximate max

        # 5. Volume Confirmation (15% weight)
        volume_score = 0
        # OBV trend gets highest weight
        volume_score += 10 if volume['obv_trend'] == '↑' else -10
        # Volume spikes with trend confirmation
        if volume['volume_spike']:
            volume_score += 5 if volume['obv_trend'] == '↑' else -5
        # ADI confirmation
        volume_score += 3 if volume['adi_trend'] == '↑' else -3
        # VWAP position
        volume_score += 2 if volume['vwap_relation'] == 'above' else -2

        # Normalize volume score to 15 points max
        score += (volume_score / 20) * 15  # 20 is max possible

        # 6. Price Action (10% weight)
        price_score = 0
        # Recent performance matters more
        price_score += (price_changes['short'] / 2) * 5  # 2% change = 5 points
        price_score += (price_changes['medium'] / 5) * 3  # 5% change = 3 points
        price_score += (price_changes['long'] / 10) * 2  # 10% change = 2 points

        # Normalize price score to 10 points max
        score += (price_score / 10) * 10  # Already scaled properly

        # Final score adjustment
        score = max(0, min(100, score))  # Ensure within bounds

        # Generate trading signal with candlestick pattern confirmation
        signal = "HOLD"
        signal_strength = ""
        
        # STRONG BUY: All PSAR timeframes bullish + high score + bullish pattern
        if (score > 75 and 
            trends['sar_short_bullish'] and 
            trends['sar_medium_bullish'] and 
            trends['sar_long_bullish'] and 
            volume['obv_trend'] == '↑' and
            (trends['bullish_engulfing'] or trends['hammer'] or trends['morning_star'])):
            signal = "STRONG BUY"
            signal_strength = "(Multi-Timeframe PSAR + Bullish Pattern)"
        
        # BUY: Partial PSAR confirmation + bullish pattern
        elif (score > 60 and 
              (trends['sar_short_bullish'] or trends['sar_medium_bullish']) and 
              volume['obv_trend'] == '↑' and
              (trends['bullish_engulfing'] or trends['hammer'])):
            signal = "BUY"
            signal_strength = "(PSAR Bullish + Pattern)"

        # STRONG SELL: All PSAR timeframes bearish + low score + bearish pattern
        elif (score < 30 and 
              trends['sar_short_bearish'] and 
              trends['sar_medium_bearish'] and 
              trends['sar_long_bearish'] and 
              volume['obv_trend'] == '↓' and
              (trends['bearish_engulfing'] or trends['shooting_star'] or trends['evening_star'])):
            signal = "STRONG SELL"
            signal_strength = "(Multi-Timeframe PSAR + Bearish Pattern)"
            
        # SELL: Partial PSAR confirmation + bearish pattern
        elif (score < 45 and 
              (trends['sar_short_bearish'] or trends['sar_medium_bearish']) and 
              volume['obv_trend'] == '↓' and
              (trends['bearish_engulfing'] or trends['shooting_star'])):
            signal = "SELL"
            signal_strength = "(PSAR Bearish + Pattern)"
        
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
            'Candle_Pattern': get_candle_pattern_name(trends),
            'Score': f"{score:.1f}",
            'Signal': signal
        }
        
        return result
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

def get_candle_pattern_name(trends):
    """Helper method to get the name of the most significant candle pattern"""
    if trends['morning_star']:
        return "Morning Star"
    if trends['evening_star']:
        return "Evening Star"
    if trends['bullish_engulfing']:
        return "Bullish Engulfing"
    if trends['bearish_engulfing']:
        return "Bearish Engulfing"
    if trends['hammer']:
        return "Hammer"
    if trends['shooting_star']:
        return "Shooting Star"
    if trends['piercing_line']:
        return "Piercing Line"
    if trends['dark_cloud_cover']:
        return "Dark Cloud Cover"
    return "None"

def calculate_and_sort_signals(tickers, short_period=14, medium_period=26, long_period=50):
    """
    Calculate signals for multiple tickers and sort them by signal strength
    Returns: DataFrame sorted by signal strength (Strong Buy -> Hold -> Strong Sell)
    """
    all_signals = []
    
    for ticker in tickers:
        signal_data = calculate_signals(ticker, short_period, medium_period, long_period)
        if signal_data is not None:
            # Add signal strength score for sorting
            signal_strength = 0
            if "STRONG BUY" in signal_data['Signal']:
                signal_strength = 4
            elif "BUY" in signal_data['Signal']:
                signal_strength = 3
            elif "HOLD" in signal_data['Signal']:
                signal_strength = 2
            elif "SELL" in signal_data['Signal']:
                signal_strength = 1
            elif "STRONG SELL" in signal_data['Signal']:
                signal_strength = 0
            
            signal_data['Signal_Strength'] = signal_strength
            all_signals.append(signal_data)
    
    if not all_signals:
        return []
    # Sort the list of dictionaries
    sorted_signals = sorted(
        all_signals,
        key=lambda x: (x['Signal_Strength'], float(x['Score'])),
        reverse=True
    )
    
    # Add ranking to each dictionary
    for rank, signal in enumerate(sorted_signals, start=1):
        signal['Rank'] = rank
    
    return sorted_signals

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
            <li><b>Score</b>: score>75 Very Strong, score>60 strong score>45 moderate score>30 weak else very weak</li>
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
        print(f"Analyzing {ticker}...")
        try:
            signal = calculate_signals(ticker, datetime.today(),short_period, medium_period, long_period)
            if signal:
                results.append(signal)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
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
    stocks = [stock for stocks in sector_stocks.values() for stock in stocks] # Replace with your stock list
    analyze_stocks(stocks, short_period=5, medium_period=10, long_period=20)