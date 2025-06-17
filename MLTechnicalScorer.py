import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
import os
from my_stocks import my_stocks
from sector_mapping import sector_stocks

warnings.simplefilter(action='ignore', category=FutureWarning)

class MLTechnicalScorer:
    def __init__(self, model_path='ta_model_2.joblib', retrain_days=30,
                 short_period=14, medium_period=26, long_period=50,
                 vix_file='vix_data.csv'):
        self.model_path = model_path
        self.retrain_days = retrain_days
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.vix_file = vix_file
        self.vix_data = None
        self.model = None
        self.last_trained = None
        self.scaler = StandardScaler()
        
        # Load and preprocess VIX data
        self._load_and_preprocess_vix_data()
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.model = load(model_path)
            self.last_trained = datetime.fromtimestamp(os.path.getmtime(model_path))

    def _load_and_preprocess_vix_data(self):
        """Load and preprocess VIX data from CSV with proper datetime parsing"""
        try:
            # Load VIX data with proper datetime parsing
            self.vix_data = pd.read_csv(
                self.vix_file,
                parse_dates=['Date '],
                date_parser=lambda x: pd.to_datetime(x, format='%d-%b-%Y')
                  # Specify your actual date format
            )

            self.vix_data.columns = [
            'date', 'vix_open', 'vix_high', 'vix_low', 'vix_close',
            'vix_prev_close', 'vix_change', 'vix_pct_change']

            
            # Set date as index with proper datetime handling
            self.vix_data['date'] = pd.to_datetime(self.vix_data['date'], format='%Y-%m-%d').dt.tz_localize(None)  # Specify format
            self.vix_data.set_index('date', inplace=True)
            # Calculate consistent VIX features
            self.vix_data['vix_range'] = self.vix_data['vix_high'] - self.vix_data['vix_low']
            self.vix_data['vix_range_pct'] = self.vix_data['vix_range'] / self.vix_data['vix_prev_close']
            
            for period in [self.short_period, self.medium_period, self.long_period]:
                ma_col = f'vix_ma_{period}'
                self.vix_data[ma_col] = self.vix_data['vix_close'].rolling(period).mean()
                self.vix_data[f'vix_ma_ratio_{period}'] = self.vix_data['vix_close'] / self.vix_data[ma_col]
                self.vix_data[f'vix_above_ma_{period}'] = (self.vix_data['vix_close'] > self.vix_data[ma_col]).astype(int)
            
            self.vix_data['vix_change_dir'] = np.where(self.vix_data['vix_change'] > 0, 1, -1)
            
            print(f"Successfully loaded VIX data from {self.vix_data.index.min().strftime('%d-%m-%Y')} "
                  f"to {self.vix_data.index.max().strftime('%d-%m-%Y')}")
        
        except Exception as e:
            print(f"Error loading VIX data: {str(e)}")
            self.vix_data = None

    def _merge_vix_data(self, df):
        """Merge VIX data with stock data using properly aligned datetimes"""
        if self.vix_data is None or len(self.vix_data) == 0:
            return df
        
        # Ensure both indices are datetime and normalized
        df.index = pd.to_datetime(df.index).tz_localize(None)  # Remove time component if any
        vix_data = self.vix_data.copy()
        vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)
        
        # Align dates and merge
        merged = df.merge(
            vix_data,
            how='left',
            left_index=True,
            right_index=True
        )
        
        # Forward fill VIX columns only
        vix_cols = list(vix_data.columns)
        merged[vix_cols] = merged[vix_cols].ffill()
        
        # Backfill any initial missing values
        merged[vix_cols] = merged[vix_cols].bfill()
    
        return merged
   
    def prepare_features(self, df, lookforward_days=5):
        """
        Prepare feature matrix with enhanced VIX features
        Returns:
            X (DataFrame): Features
            y (Series): Target (future returns)
        """
        # Merge VIX data
        df = self._merge_vix_data(df)
        #print(df.columns)
        # Calculate all technical indicators
        indicators = self._calculate_all_indicators(df)
        
        # Create features DataFrame
        features = pd.DataFrame()
        
        # Price features
        features['price'] = df['Close']
        features[f'returns_{self.short_period}d'] = df['Close'].pct_change(self.short_period)
        features[f'returns_{self.medium_period}d'] = df['Close'].pct_change(self.medium_period)
        features[f'returns_{self.long_period}d'] = df['Close'].pct_change(self.long_period)
        
        # Add VIX features with consistent naming
        if 'vix_close' in df.columns:
            features['vix'] = df['vix_close']
            features['vix_pct_change'] = df['vix_pct_change']
            features['vix_range'] = df['vix_range']
            
            for period in [self.short_period, self.medium_period, self.long_period]:
                features[f'vix_ma_ratio_{period}'] = df[f'vix_ma_ratio_{period}']
                features[f'vix_above_ma_{period}'] = df[f'vix_above_ma_{period}']
            
            features['vix_change_dir'] = df['vix_change_dir']
            
            # VIX momentum
            features['vix_change_direction'] = np.where(df['vix_change'] > 0, 1, -1)
            features['vix_range_pct'] = df['vix_high'] - df['vix_low'] / df['vix_prev_close']
        
        # Add all technical indicators
        for name, values in indicators.items():
            if isinstance(values, (pd.Series, pd.DataFrame)):
                features[name] = values
            
        # Calculate target (future returns)
        features['target'] = df['Close'].pct_change(lookforward_days).shift(-lookforward_days)
        
        # Drop rows with missing values
        features = features.dropna()
        
        if len(features) == 0:
            return None, None
            
        X = features.drop('target', axis=1)
        y = features['target']
        
        return X, y
    
    def _calculate_all_indicators(self, df):
        """Calculate all technical indicators for feature generation"""
        indicators = {}
        
        # Momentum indicators for all timeframes
        for period, suffix in [(self.short_period, 'short'), 
                             (self.medium_period, 'medium'), 
                             (self.long_period, 'long')]:
            indicators[f'rsi_{suffix}'] = RSIIndicator(close=df['Close'], window=period).rsi()
            indicators[f'roc_{suffix}'] = ROCIndicator(close=df['Close'], window=period).roc()
            
            # Stochastic only for short timeframe
            if suffix == 'short':
                indicators[f'stoch_{suffix}'] = StochasticOscillator(
                    high=df['High'], low=df['Low'], close=df['Close'], 
                    window=period, smooth_window=3).stoch()
        
        # Trend indicators
        for period, suffix in [(self.short_period, 'short'), 
                             (self.medium_period, 'medium'), 
                             (self.long_period, 'long')]:
            indicators[f'sma_{suffix}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
            indicators[f'ema_{suffix}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            
            # MACD only for medium timeframe (standard settings)
            if suffix == 'medium':
                macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
                indicators['macd'] = macd.macd()
                indicators['macd_diff'] = macd.macd_diff()
                
            # ADX only for medium timeframe
            if suffix == 'medium':
                indicators['adx'] = ADXIndicator(
                    high=df['High'], low=df['Low'], close=df['Close'], window=period).adx()
        
        # Volatility indicators
        for period, suffix in [(self.short_period, 'short'), 
                             (self.medium_period, 'medium'), 
                             (self.long_period, 'long')]:
            bb = BollingerBands(close=df['Close'], window=period, window_dev=2)
            indicators[f'bb_upper_{suffix}'] = bb.bollinger_hband()
            indicators[f'bb_middle_{suffix}'] = bb.bollinger_mavg()
            indicators[f'bb_lower_{suffix}'] = bb.bollinger_lband()
            
            # ATR for all timeframes
            indicators[f'atr_{suffix}'] = AverageTrueRange(
                high=df['High'], low=df['Low'], close=df['Close'], window=period).average_true_range()
        
        # Volume indicators
        for period, suffix in [(self.short_period, 'short'), 
                             (self.medium_period, 'medium'), 
                             (self.long_period, 'long')]:
            indicators[f'vwap_{suffix}'] = VolumeWeightedAveragePrice(
                high=df['High'], low=df['Low'], close=df['Close'], 
                volume=df['Volume'], window=period).volume_weighted_average_price()
        
        # Volume indicators that don't use timeframes
        indicators['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        indicators['adi'] = AccDistIndexIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()
        indicators['volume_ma'] = df['Volume'].rolling(self.short_period).mean()
        indicators['volume_spike'] = (df['Volume'] > (df['Volume'].rolling(self.short_period).mean() * 2)).astype(int)
        
        # PSAR indicators for all timeframes with different parameters
        psar_params = {
            'short': (0.02, 0.2),
            'medium': (0.015, 0.15),
            'long': (0.01, 0.1)
        }
        
        for suffix, (step, max_step) in psar_params.items():
            period = getattr(self, f'{suffix}_period')
            psar = PSARIndicator(
                high=df['High'], low=df['Low'], close=df['Close'],
                step=step, max_step=max_step).psar()
            indicators[f'psar_{suffix}'] = psar
            indicators[f'psar_bullish_{suffix}'] = (df['Close'] > psar).astype(int)
        
        return indicators
    
    def train_model(self, tickers, start_date, end_date):
        """Train model on historical data with enhanced VIX features"""
        all_X = []
        all_y = []
        print("training")
        for ticker in tickers:
            try:
                # Download historical data - need enough for longest timeframe
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date - timedelta(days=self.long_period*3),
                                  end=end_date, 
                                  interval='1d')
                #print(len(df), self.long_period * 3)
                if len(df) < self.long_period * 3:  # Need at least 3x longest period
                    continue
                    
                # Prepare features with VIX
                X, y = self.prepare_features(df)
                #print("Train columns")
                if X is not None and y is not None:
                    all_X.append(X)
                    all_y.append(y)
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        if not all_X:
            raise ValueError("No valid data found for training")
            
        # Combine all data
        X_combined = pd.concat(all_X)
        y_combined = pd.concat(all_y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42)
        
        # Create pipeline with scaling and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1))
        ])

        # Train model
        #print(X_train.columns)
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        print(f"Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        # Show top feature importances
        print("\nTop 20 Feature Importances:")
        feature_importances = pd.Series(
            pipeline.named_steps['model'].feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        print(feature_importances.head(20))
        
        # Save model
        self.model = pipeline
        self.last_trained = datetime.now()
        dump(pipeline, self.model_path)
        return pipeline

    def should_retrain(self):
        """Check if model needs retraining"""
        if self.last_trained is None:
            return True
        return (datetime.now() - self.last_trained).days >= self.retrain_days
    
    def calculate_score_and_signals(self, ticker, current_date):
        """Calculate ML-based score and generate trading signals with VIX"""
        if self.model is None or self.should_retrain():
            print("Model not loaded or needs retraining - training now...")
            self.train_model([stock for stocks in sector_stocks.values() for stock in stocks], current_date - timedelta(days=365*3),current_date)

        
        try:
            # Download data - need enough for longest timeframe
            end_date = current_date
            start_date = end_date - timedelta(days=self.long_period*3)
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval='1d')
            if len(df) < self.long_period:  # Minimum data required
                return None
            
            # Prepare features with VIX
            X, _ = self.prepare_features(df)
            if X is None or len(X) == 0:
                return None
            #print(len(X))
            #X.to_csv('test_1.csv', index=False)
            # Get most recent features
            latest_features = X.iloc[-1:].copy()
            # Make prediction
            predicted_return = self.model.predict(latest_features)[0]
            
            # Convert return prediction to score (0-100)
            score = 50 + (predicted_return * 500)  # Scale to make meaningful differences
            score = max(0, min(100, score))  # Clamp to 0-100
            
            # Generate signals based on indicators, score, and VIX
            signals = self._generate_signals(df, latest_features.iloc[0], score)
            
            # Prepare result dictionary
            result = {
                'Ticker': ticker,
                'Date': current_date.strftime('%Y-%m-%d'),
                'Price': f"{df['Close'].iloc[-1]:.2f}",
                f'Change_{self.short_period}D': f"{df['Close'].iloc[-1]/df['Close'].iloc[-self.short_period-1]-1:.1%}",
                f'Change_{self.medium_period}D': f"{df['Close'].iloc[-1]/df['Close'].iloc[-self.medium_period-1]-1:.1%}",
                f'Change_{self.long_period}D': f"{df['Close'].iloc[-1]/df['Close'].iloc[-self.long_period-1]-1:.1%}",
                'VIX': f"{latest_features.get('vix', [None])[0]:.2f}" if 'vix' in latest_features else 'N/A',
                'VIX_%_Change': f"{latest_features.get('vix_pct_change', [None])[0]:.1%}" if 'vix_pct_change' in latest_features else 'N/A',
                f'VIX_MA_Ratio_{self.medium_period}': f"{latest_features.get(f'vix_ma_ratio_{self.medium_period}', [None])[0]:.2f}" if f'vix_ma_ratio_{self.medium_period}' in latest_features else 'N/A',
                f'RSI_{self.short_period}': f"{latest_features[f'rsi_short'].iloc[0]:.1f}",
                f'RSI_{self.medium_period}': f"{latest_features[f'rsi_medium'].iloc[0]:.1f}",
                f'Stoch_%K_{self.short_period}': f"{latest_features[f'stoch_short'].iloc[0]:.1f}",
                f'MACD_diff_{self.medium_period}': f"{latest_features['macd_diff'].iloc[0]:.3f}",
                'BB_%_short': f"{(df['Close'].iloc[-1] - latest_features['bb_lower_short'].iloc[0])/(latest_features['bb_upper_short'].iloc[0] - latest_features['bb_lower_short'].iloc[0]):.1%}",
                f'SMA_{self.short_period}/{self.medium_period}/{self.long_period}': f"{latest_features[f'sma_short'].iloc[0]:.1f}/{latest_features[f'sma_medium'].iloc[0]:.1f}/{latest_features[f'sma_long'].iloc[0]:.1f}",
                f'PSAR_{self.short_period}': 'Bullish' if latest_features['psar_bullish_short'].iloc[0] else 'Bearish',
                f'PSAR_{self.medium_period}': 'Bullish' if latest_features['psar_bullish_medium'].iloc[0] else 'Bearish',
                f'PSAR_{self.long_period}': 'Bullish' if latest_features['psar_bullish_long'].iloc[0] else 'Bearish',
                'Volume': f"{df['Volume'].iloc[-1]/1e6:.1f}M",
                'Volume_Spike': 'Yes' if latest_features['volume_spike'].iloc[0] else 'No',
                'OBV_Trend': '↑' if latest_features['obv'].iloc[0] > X['obv'].iloc[-2] else '↓',
                'VWAP_Relation': 'above' if df['Close'].iloc[-1] > latest_features['vwap_medium'].iloc[0] else 'below',
                'Score': f"{score:.1f}",
                'Predicted_Return': f"{predicted_return:.2%}",
                'Signal': signals['primary'],
                'Signal_Reasons': signals['reasons'],
                'VIX_Context': signals['vix_context']
            }
            
            return result
            
        except Exception as e:
            print(f"Error calculating score for {ticker}: {str(e)}")
            return None
    
    def _generate_signals(self, df, features, score):
        """Generate trading signals considering all VIX features"""
        signals = {
            'primary': 'HOLD',
            'reasons': [],
            'vix_context': None
        }
        
        # Get VIX values if available
        vix = features.get('vix')
        vix_pct_change = features.get('vix_pct_change')
        vix_ma_ratio = features.get(f'vix_ma_ratio_{self.medium_period}')
        vix_range = features.get('vix_range')
        
        # VIX context analysis
        vix_context = []
        if vix is not None:
            if vix > 30:
                vix_context.append("High VIX (>30)")
            elif vix < 15:
                vix_context.append("Low VIX (<15)")
                
            if vix_ma_ratio is not None:
                if vix_ma_ratio > 1.2:
                    vix_context.append(f"VIX {vix_ma_ratio:.1f}x above MA")
                elif vix_ma_ratio < 0.8:
                    vix_context.append(f"VIX {vix_ma_ratio:.1f}x below MA")
            
            if vix_pct_change is not None and abs(vix_pct_change) > 0.1:
                direction = "up" if vix_pct_change > 0 else "down"
                vix_context.append(f"VIX {direction} {abs(vix_pct_change):.2f}%")
        
        signals['vix_context'] = ", ".join(vix_context) if vix_context else "Normal"
        
        # Get PSAR states
        psar_short = features['psar_bullish_short']
        psar_medium = features['psar_bullish_medium']
        psar_long = features['psar_bullish_long']
        
        # Get trend states
        price = df['Close'].iloc[-1]
        sma_short = features['sma_short']
        sma_medium = features['sma_medium']
        sma_long = features['sma_long']
        
        # Volume indicators
        obv_trend = '↑' if features['obv'] > features['obv'] else '↓'  # Simplified
        volume_spike = features['volume_spike']
        
        # Generate reasons
        if score > 70:
            signals['reasons'].append(f"High ML Score ({score:.1f})")
        if score < 30:
            signals['reasons'].append(f"Low ML Score ({score:.1f})")
            
        if psar_short and psar_medium and psar_long:
            signals['reasons'].append("PSAR Bullish All Timeframes")
        elif psar_short or psar_medium:
            signals['reasons'].append("PSAR Bullish Some Timeframes")
            
        if price > sma_short > sma_medium > sma_long:
            signals['reasons'].append("Strong Uptrend (Price > SMAs)")
            
        if obv_trend == '↑' and volume_spike:
            signals['reasons'].append("OBV Up with Volume Spike")
        
        # VIX-based adjustments to signals
        vix_modifier = ""
        if vix is not None:
            if vix > 30:
                vix_modifier = " (High VIX Caution)"
            elif vix < 15:
                vix_modifier = " (Low VIX Opportunity)"
            
            if vix_ma_ratio is not None and vix_ma_ratio > 1.5:
                vix_modifier = " (Extreme VIX Caution)"
        
        # Generate primary signal with VIX context
        if (score > 75 and psar_short and psar_medium and psar_long and 
            price > sma_short > sma_medium > sma_long):
            signals['primary'] = f"STRONG BUY{vix_modifier}"
        elif score > 65 and (psar_short or psar_medium) and price > sma_short:
            signals['primary'] = f"BUY{vix_modifier}"
        elif (score < 25 and not psar_short and not psar_medium and not psar_long and 
              price < sma_short < sma_medium):
            signals['primary'] = f"STRONG SELL{vix_modifier}"
        elif score < 35 and not (psar_short and psar_medium) and price < sma_short:
            signals['primary'] = f"SELL{vix_modifier}"
        
        return signals

# Example usage:
if __name__ == "__main__":
    # Initialize with path to your VIX CSV file
    scorer = MLTechnicalScorer(
        short_period=14,
        medium_period=26,
        long_period=50,
        vix_file='hist_india_vix_-18-06-2024-to-17-06-2025.csv'  # Your CSV with all VIX fields
    )
    
    # Calculate score and signals for a stock
    result = scorer.calculate_score_and_signals('ZEEL.NS', datetime.now())
    
    if result:
        print("\nTechnical Analysis Results with Enhanced VIX:")
        print("-" * 60)
        for key, value in result.items():
            if key == 'features':
                continue
            print(f"{key:>25}: {value}")