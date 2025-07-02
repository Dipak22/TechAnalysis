import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator
from datetime import datetime, timedelta
from sector_mapping import sector_stocks  # Replace with your stock list
from my_stocks import my_stocks, PENNY_STOCKS,NEW_STOCKS,SHORT_TERM_STOCKS, CASH_HEAVY

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
            ).psar(),
            'ADX': ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=short_period).adx()
        }

        # Medium-term indicators (26 days)
        adx_indicator = ADXIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=medium_period
        )
        macd_indicator = MACD(
            close=df['Close'],
            window_slow=20,
            window_fast=6,
            window_sign=10
        )
        indicators['medium'] = {
            'RSI': RSIIndicator(close=df['Close'], window=medium_period).rsi(),
            'ROC': ROCIndicator(close=df['Close'], window=medium_period).roc(),
            'SMA': SMAIndicator(close=df['Close'], window=medium_period).sma_indicator(),
            'EMA': EMAIndicator(close=df['Close'], window=medium_period).ema_indicator(),
            'MACD': macd_indicator.macd(),
            'MACD_hist': macd_indicator.macd_diff(),
            'MACD_signal': macd_indicator.macd_signal(),
            'ADX': adx_indicator.adx(),
            'DMP': adx_indicator.adx_pos(),  # Positive Directional Movement
            'DMN': adx_indicator.adx_neg(),  # Negative Directional Movement
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
            ).psar(),
            'ADX': ADXIndicator(high= df['High'], low=df['Low'], close=df['Close'], window=long_period).adx()
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
            'macd_bullish': latest['medium']['MACD_hist'] > prev['medium']['MACD_hist'] and latest['medium']['MACD'] > latest['medium']['MACD_signal'] and latest['medium']['MACD'] > prev['medium']['MACD'] and latest['medium']['MACD'] > 0,
            'macd_bearish': latest['medium']['MACD_hist'] < prev['medium']['MACD_hist'] and latest['medium']['MACD'] < latest['medium']['MACD_signal'] and latest['medium']['MACD'] < prev['medium']['MACD'] and latest['medium']['MACD'] < 0,
            'adx_strength': latest['medium']['ADX'] > 25,
            'dmp_dominant': latest['medium']['DMP'] > latest['medium']['DMN'],  # Positive directional movement
            'dmn_dominant': latest['medium']['DMN'] > latest['medium']['DMP'],  # Negative directional movement
            'adx_bullish': latest['short']['ADX'] > latest['medium']['ADX'] and latest['medium']['ADX']>latest['long']['ADX'],  # ADX trending up
            'adx_bearish': latest['short']['ADX'] < latest['medium']['ADX'] and latest['medium']['ADX']<latest['long']['ADX'],  # ADX trending down
            
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
            signal_reasons.append(f"Strong Trend (ADX {latest['medium']['ADX']:.1f})")
        if trends['dmp_dominant']:
            signal_reasons.append(f"Bullish DMI (DMP {latest['medium']['DMP']:.1f} > DMN {latest['medium']['DMN']:.1f})")
        if trends['dmn_dominant']:
            signal_reasons.append(f"Bearish DMI (DMN {latest['medium']['DMN']:.1f} > DMP {latest['medium']['DMP']:.1f})")

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
            
        # adx reasons
        if trends['adx_bullish']:
            signal_reasons.append("ADX Bullish Trend (Short > Medium > Long)")
        if trends['adx_bearish']:
            signal_reasons.append("ADX Bearish Trend (Short < Medium < Long)")
        
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

        # ADX and DMI components
        adx_component = min(50, latest['medium']['ADX']) / 2  # Normalize ADX (0-25 points)
        trend_score += adx_component if trends['adx_strength'] else 0
        trend_score += 5 if trends['dmp_dominant'] else (-5 if trends['dmn_dominant'] else 0)
        

        # Normalize trend score to 30 points max
        score += (trend_score / 67) * 30  # 67 is max possible trend score

        # 2. PSAR Confirmation (20% weight)
        psar_score = 0
        # Weighted by timeframe importance
        psar_score += 5 if trends['sar_short_bullish'] else -5
        psar_score += 7 if trends['sar_medium_bullish'] else -7
        psar_score += 8 if trends['sar_long_bullish'] else -8

        # Normalize PSAR score to 20 points max
        score += (psar_score / 20) * 20  # 20 is max possible PSAR score

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

        # Normalize pattern score to 15 points max
        pattern_score = max(-15, min(15, pattern_score))
        score += pattern_score

         # 4. Momentum Strength (20% weight)
        momentum_score = 0
        # RSI - parabolic weighting (more extreme = stronger signal)
        rsi_weight = 0
        if momentum['rsi_short'] > 70:
            rsi_weight = -((momentum['rsi_short'] - 70) / 30) ** 2 * 10  # Overbought penalty
        elif momentum['rsi_short'] < 30:
            rsi_weight = ((30 - momentum['rsi_short']) / 30) ** 2 * 10  # Oversold bonus
        momentum_score += rsi_weight

        # Stochastic - similar parabolic weighting
        stoch_weight = 0
        if momentum['stoch_overbought']:
            stoch_weight = -((latest['short']['Stoch_%K'] - 80) / 20) ** 2 * 5
        elif momentum['stoch_oversold']:
            stoch_weight = ((20 - latest['short']['Stoch_%K']) / 20) ** 2 * 5
        momentum_score += stoch_weight

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

        # Final score adjustment
        score = max(0, min(100, score))  # Ensure within bounds

        # Generate trading signal with candlestick pattern confirmation
        signal = "HOLD"
        signal_strength = ""
        signal_value = 0
        
        # STRONG BUY: All PSAR timeframes bullish + high score + bullish pattern + DMP dominant + MACD bullish
        if (trends['adx_bullish'] and
            trends['sar_short_bullish'] and 
            trends['sar_medium_bullish'] and 
            trends['sar_long_bullish'] and 
            volume['obv_trend'] == '↑' and
            trends['dmp_dominant'] and
            trends['macd_bullish']):
            signal = "STRONG BUY"
            signal_strength = "(Multi-Timeframe PSAR + Bullish Pattern + DMP + MACD)"
            signal_value = 6

        # BUY: Partial PSAR confirmation + bullish pattern + DMP + MACD bullish
        elif (trends['adx_bullish'] and
              (trends['sar_short_bullish'] or trends['sar_medium_bullish']) and 
              volume['obv_trend'] == '↑' and
              trends['dmp_dominant'] and
              trends['macd_bullish'] ):
            signal = "BUY"
            signal_strength = "(PSAR Bullish + Pattern + DMP + MACD)"
            signal_value = 5

        # STRONG SELL: All PSAR timeframes bearish + low score + bearish pattern + DMN + MACD bearish
        elif (trends['adx_bearish'] and
              trends['sar_short_bearish'] and 
              trends['sar_medium_bearish'] and 
              trends['sar_long_bearish'] and 
              volume['obv_trend'] == '↓' and
              trends['dmn_dominant'] and
              trends['macd_bearish']):
            signal = "STRONG SELL"
            signal_strength = "(Multi-Timeframe PSAR + Bearish Pattern + DMN + MACD)"
            signal_value = 1
            
        # SELL: Partial PSAR confirmation + bearish pattern + DMN + MACD bearish
        elif (trends['adx_bearish'] and
              (trends['sar_short_bearish'] or trends['sar_medium_bearish']) and 
              volume['obv_trend'] == '↓' and
              trends['dmn_dominant'] and
              trends['macd_bearish']):
            signal = "SELL"
            signal_strength = "(PSAR Bearish + Pattern + DMN + MACD)"
            signal_value = 2
            
        # WEAK BUY/SELL signals (less strict conditions)
        elif ( #score > 60 and 
              trends['macd_bullish'] and
              (trends['sar_short_bullish'] or trends['sar_medium_bullish'])):
            signal = "WEAK BUY"
            signal_strength = "(MACD Bullish + Partial PSAR)"
            signal_value = 4
            
        elif (#score < 40 and 
              trends['macd_bearish'] and
              (trends['sar_short_bearish'] or trends['sar_medium_bearish'])):
            signal = "WEAK SELL"
            signal_strength = "(MACD Bearish + Partial PSAR)"
            signal_value = 3

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
            f'MACD_diff_{medium_period}': 'Bullish' if trends['macd_bullish'] else ('Bearish' if trends['macd_bearish'] else 'Neutral'), 
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
            'Signal': signal,
            'Signal_Value': signal_value
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

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# [Previous code remains the same until generate_html_report function]

def generate_html_report(short_period, medium_period, long_period, results, output_file='momentum_report.html'):
    """Generate HTML report with dynamic period headers and interactive ticker links"""
    
    # Generate plot images for each stock first
    plot_images = {}
    for result in results:
        ticker = result['Ticker']
        plot_images[ticker] = generate_stock_plot(ticker, short_period, medium_period, long_period)
    
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
            .ticker-link {{
                color: #0066cc;
                text-decoration: none;
                font-weight: bold;
            }}
            .ticker-link:hover {{
                text-decoration: underline;
            }}
        </style>
        <script>
            function openStockChart(ticker) {{
                window.open('stock_chart_' + ticker + '.html', '_blank');
            }}
        </script>
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
                    <td><a href="#" onclick="openStockChart('{r['Ticker']}')" class="ticker-link">{r['Ticker']}</a></td>
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
                    <td class="{{'positive' if r[f'MACD_diff_{medium_period}'] == 'Bullish' else 'negative'}}">
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
    
    # Generate individual stock chart pages
    for ticker, plot_html in plot_images.items():
        with open(f'stock_chart_{ticker}.html', 'w') as f:
            f.write(plot_html)
    
    print(f"Report generated: {output_file}")

def generate_stock_plot(ticker, short_period, medium_period, long_period):
    """Generate a plot of key indicators for a stock and return as HTML"""
    try:
        # Download data
        end_date = datetime.today()
        start_date = end_date - timedelta(days=long_period*3)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty or len(df) < long_period:
            return "<p>No data available for plotting</p>"
        
        # Calculate indicators
        # Moving Averages
        sma_short = SMAIndicator(close=df['Close'], window=short_period).sma_indicator()
        sma_medium = SMAIndicator(close=df['Close'], window=medium_period).sma_indicator()
        sma_long = SMAIndicator(close=df['Close'], window=long_period).sma_indicator()
        ema_short = EMAIndicator(close=df['Close'], window=short_period).ema_indicator()
        ema_medium = EMAIndicator(close=df['Close'], window=medium_period).ema_indicator()
        
        # Momentum Indicators
        short_rsi = RSIIndicator(close=df['Close'], window=short_period).rsi()
        medium_rsi = RSIIndicator(close=df['Close'], window=medium_period).rsi()
        macd = MACD(close=df['Close'], window_slow=20, window_fast=6, window_sign=10).macd()
        macd_signal = MACD(close=df['Close'], window_slow=20, window_fast=6, window_sign=10).macd_signal()
        macd_hist = MACD(close=df['Close'], window_slow=20, window_fast=6, window_sign=10).macd_diff()
        
        # Trend Indicators
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=medium_period)
        psar_short = FixedPSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2).psar()
        psar_medium = FixedPSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.015, max_step=0.15).psar()
        
        # Volatility Indicators
        bb = BollingerBands(close=df['Close'], window=long_period, window_dev=2)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=medium_period).average_true_range()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 18), dpi=100)
        gs = fig.add_gridspec(6, 1, height_ratios=[3, 2, 2, 2, 2, 1])
        
        # Plot 1: Price with Moving Averages and PSAR
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, df['Close'], label='Price', color='blue', linewidth=2)
        ax1.plot(df.index, sma_short, label=f'SMA {short_period}D', color='orange', alpha=0.7)
        ax1.plot(df.index, sma_medium, label=f'SMA {medium_period}D', color='green', alpha=0.7)
        ax1.plot(df.index, sma_long, label=f'SMA {long_period}D', color='purple', alpha=0.7)
        ax1.plot(df.index, ema_short, label=f'EMA {short_period}D', color='orange', linestyle='--', alpha=0.7)
        ax1.plot(df.index, ema_medium, label=f'EMA {medium_period}D', color='green', linestyle='--', alpha=0.7)
        ax1.plot(df.index, psar_short, 'r.', label=f'PSAR {short_period}D', markersize=4)
        ax1.plot(df.index, psar_medium, 'g.', label=f'PSAR {medium_period}D', markersize=4)
        ax1.set_title(f'{ticker} Price with Moving Averages and PSAR')
        ax1.legend(loc='upper left')
        
        # Plot 2: Bollinger Bands
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.5)
        ax2.plot(df.index, bb.bollinger_hband(), label='Upper Band', color='green', linestyle='--')
        ax2.plot(df.index, bb.bollinger_mavg(), label='Middle Band', color='black', linestyle='--')
        ax2.plot(df.index, bb.bollinger_lband(), label='Lower Band', color='red', linestyle='--')
        ax2.fill_between(df.index, bb.bollinger_hband(), bb.bollinger_lband(), color='gray', alpha=0.1)
        bb_percent = ((df['Close'] - bb.bollinger_lband()) / 
                     (bb.bollinger_hband() - bb.bollinger_lband()))
        ax2b = ax2.twinx()
        ax2b.plot(df.index, bb_percent, label='BB %', color='purple', alpha=0.3)
        ax2b.set_ylim(0, 1)
        ax2b.axhline(0.8, color='red', linestyle=':', alpha=0.3)
        ax2b.axhline(0.2, color='green', linestyle=':', alpha=0.3)
        ax2.set_title('Bollinger Bands')
        ax2.legend(loc='upper left')
        ax2b.legend(loc='upper right')
        
        # Plot 3: ADX and DMI
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax3.plot(df.index, adx.adx(), label='ADX', color='black')
        ax3.plot(df.index, adx.adx_pos(), label='+DMI', color='green')
        ax3.plot(df.index, adx.adx_neg(), label='-DMI', color='red')
        ax3.axhline(25, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('ADX with DMI')
        ax3.legend()
        
        # Plot 4: RSI
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        ax4.plot(df.index, short_rsi, label=f'RSI {short_period}D', color='blue')
        ax4.plot(df.index, medium_rsi, label=f'RSI {medium_period}D', color='green')
        ax4.axhline(70, color='red', linestyle='--')
        ax4.axhline(30, color='green', linestyle='--')
        ax4.set_ylim(0, 100)
        ax4.set_title('Relative Strength Index')
        ax4.legend()
        
        # Plot 5: MACD
        ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
        ax5.plot(df.index, macd, label='MACD', color='blue')
        ax5.plot(df.index, macd_signal, label='Signal', color='orange')
        ax5.bar(df.index, macd_hist, label='Histogram', color=np.where(macd_hist > 0, 'g', 'r'), alpha=0.5)
        ax5.axhline(0, color='black', linestyle='--')
        ax5.set_title('MACD')
        ax5.legend()
        
        # Plot 6: Volume
        ax6 = fig.add_subplot(gs[5, 0], sharex=ax1)
        ax6.bar(df.index, df['Volume'], label='Volume', color='blue', alpha=0.5)
        ax6.plot(df.index, df['Volume'].rolling(20).mean(), label='20D MA', color='red')
        ax6.set_title('Volume')
        ax6.legend()
        
        plt.tight_layout()
        
        # Save plot to HTML
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        html = f"""
        <html>
        <head>
            <title>{ticker} Technical Indicators</title>
            <style>
                body {{ font-family: Arial; margin: 20px; }}
                h1 {{ color: #333366; }}
                .plot {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>{ticker} Technical Indicators ({datetime.today().strftime('%Y-%m-%d')})</h1>
            <div class="plot">
                <img src="data:image/png;base64,{data}" alt="{ticker} technical indicators" style="width:100%;">
            </div>
            <div>
                <h3>Analysis Periods:</h3>
                <ul>
                    <li>Short-term: {short_period} days</li>
                    <li>Medium-term: {medium_period} days</li>
                    <li>Long-term: {long_period} days</li>
                </ul>
                <h3>Indicators Shown:</h3>
                <ul>
                    <li><strong>Price with Moving Averages:</strong> SMA {short_period}D, {medium_period}D, {long_period}D and EMA {short_period}D, {medium_period}D</li>
                    <li><strong>Bollinger Bands:</strong> {long_period}D with 2 standard deviations</li>
                    <li><strong>ADX/DMI:</strong> {medium_period}D period showing trend strength and direction</li>
                    <li><strong>RSI:</strong> {short_period}D and {medium_period}D periods</li>
                    <li><strong>MACD:</strong> (6,20,10) settings showing momentum</li>
                    <li><strong>Volume:</strong> With 20D moving average</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    except Exception as e:
        print(f"Error generating plot for {ticker}: {str(e)}")
        return f"<p>Error generating plot for {ticker}</p>"

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
    results.sort(key=lambda x: (x['Signal_Value'], float(x['Score'])), reverse=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_FILE = f"momentum_report_CASH_stocks_{current_date}.html"
    generate_html_report(short_period,medium_period, long_period, results, output_file=OUTPUT_FILE)

# Example usage
if __name__ == "__main__":
    #stocks = [stock for stocks in sector_stocks.values() for stock in stocks] # Replace with your stock list
    stocks = CASH_HEAVY  # Example stock list
    #stocks.extend(PENNY_STOCKS)
    ##stocks.extend(NEW_STOCKS)
    
    analyze_stocks(stocks, short_period=9, medium_period=16, long_period=26)