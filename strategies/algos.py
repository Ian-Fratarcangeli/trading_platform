import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.strategy import StrategyBase
import ta
import os


class MACD(StrategyBase):
    def __init__(self, symbols, window_slow=30, window_fast=10):
        self.states = {symbol: 0 for symbol in symbols}
        self.window_slow = window_slow
        self.window_fast = window_fast


    def data_process(self, data):
        #calculate MACD and the 9-day EMA signal
        data = data.sort_values(['symbol', 'timestamp'])

        #calculate mean, std, and z-score over window
        data['macd'] = data.groupby('symbol')['price'].transform(lambda x: ta.trend.MACD(close=x.squeeze(), window_slow=self.window_slow, window_fast=self.window_fast).macd())
        data['ema'] = data.groupby('symbol')['price'].transform(lambda x: ta.trend.MACD(close=x.squeeze(), window_slow=self.window_slow, window_fast=self.window_fast).macd_signal())
        data['rsi'] = data.groupby('symbol')['price'].transform(lambda x: ta.momentum.RSIIndicator(x).rsi())
        data['return'] = data.groupby('symbol')['price'].transform(lambda x: np.log(x / x.shift(1)))
        data = data.dropna()

        data = data.sort_index()

        # Get each symbol's first available date
        first_dates = data.groupby('symbol').apply(lambda g: g.index.min())

        #common start date
        common_start = first_dates.max()
        
        data_aligned = data[data.index >= common_start]
        return data_aligned

    def data_visual(self, data):
         # Group by symbol and plot for each one
        for filename in os.listdir('data_images'):
            file_path = os.path.join('data_images', filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for symbol, symbol_data in data.groupby('symbol'):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

            # --- Price plot ---
            ax1.plot(symbol_data.index, symbol_data['price'], label='Price', color='blue')
            ax1.set_ylabel('Price')
            ax1.set_title(f'{symbol} - Price and MACD')
            ax1.legend(loc='upper left')

            # --- MACD + Signal Line plot ---
            ax2.plot(symbol_data.index, symbol_data['macd'], label='MACD', color='red')
            ax2.plot(symbol_data.index, symbol_data['ema'], label='Signal Line', color='orange')
            ax2.axhline(0, color='black', linestyle='--', linewidth=1)
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax3 = ax2.twinx()
            ax3.plot(symbol_data.index, symbol_data['rsi'], label='RSI', color='green')
            ax3.axhline(70, color='blue', linestyle='--', linewidth=1)
            ax3.axhline(30, color='blue', linestyle='--', linewidth=1)
            ax3.set_ylabel('RSI')
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper right')

            plt.xlabel('Time')
            plt.tight_layout()

            plt.savefig(f"data_images/{symbol}")
            plt.close(fig)

    def generate_signals(self, point):
        symbol = point['symbol']
        state = self.states[symbol]
        macd = point['macd']
        ema = point['ema']
        rsi = point['rsi']

        #long if rsi signals underbought, exit if rsi signals overbought
        if rsi >= 70:
            if state == 1:
                self.states[symbol] = 0
                return 'sell'     
        elif rsi <= 30:
            if state == 0:
                self.states[symbol] = 1
                return 'buy'
            
         #exit if macd crosses under, long if macd crosses over
        elif macd > ema:
            if state == 0:
                self.states[symbol] = 1
                return 'buy'
        elif macd < ema:
            if state == 1:
                self.states[symbol] = 0
                return 'sell'
        #hold otherwise
        return 'hold'

class MeanReversion(StrategyBase):
    def __init__(self, symbols, enter_threshold=1.4, exit_threshold=0.2, window=10):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.window = window
        #depicts the positional state of each equity - -1 for short, 1 for long, 0 for no position
        self.states = {symbol: 0 for symbol in symbols}

    
    def data_process(self, data):
        #time-order within symbol
        data = data.sort_values(['symbol', 'timestamp'])

        #calculate mean, std, and z-score over window
        data['rolling_mean'] = data.groupby('symbol')['price'].transform(lambda x: x.rolling(self.window).mean())
        data['rolling_std'] = data.groupby('symbol')['price'].transform(lambda x: x.rolling(self.window).std())
        data['return'] = data.groupby('symbol')['price'].transform(lambda x: np.log(x / x.shift(1)))
        # Z-score
        data['z_score'] = (data['price'] - data['rolling_mean']) / data['rolling_std']
        
        data = data.dropna()

        data = data.sort_index()

        # Get each symbol's first available date
        first_dates = data.groupby('symbol').apply(lambda g: g.index.min())

        #common start date
        common_start = first_dates.max()
        data_aligned = data[data.index >= common_start]

        return data_aligned
    
    def data_visual(self, data):
        # Group by symbol and plot for each one
        for filename in os.listdir('data_images'):
            file_path = os.path.join('data_images', filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for symbol, symbol_data in data.groupby('symbol'):
            fig, ax1 = plt.subplots(figsize=(14, 6))

            # Price plot (left y-axis)
            ax1.plot(symbol_data.index, symbol_data['price'], color='blue', label='Price')
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # Z-score plot (right y-axis)
            ax2 = ax1.twinx()
            ax2.plot(symbol_data.index, symbol_data['z_score'], color='red', label='Z-score')
            ax2.axhline(0, color='black', linewidth=1, linestyle='--')
            ax2.axhline(1, color='gray', linewidth=0.5, linestyle='--')
            ax2.axhline(-1, color='gray', linewidth=0.5, linestyle='--')
            ax2.set_ylabel('Z-score', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Title and legend
            fig.suptitle(f'{symbol} - Price and Rolling Z-Score (window={self.window})', fontsize=16)
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

            plt.savefig(f"data_images/{symbol}")
            plt.close(fig)

        
    def generate_signals(self, point):
        symbol = point['symbol']
        state = self.states[symbol]
        z = point['z_score']

        #short if z gets too high
        if z > self.enter_threshold:
            if state == 0:
                self.states[symbol] = -1
                return 'short'

        #buy if z dips too low
        elif z < -self.enter_threshold:
            if state == 0:
                self.states[symbol] = 1
                return 'buy'
    
        #situations when z gets to mean or were to overtake the exit threshold but still cross the mean    
        elif (z > self.exit_threshold) or (np.abs(z) < self.exit_threshold):
            if state == 1:
                self.states[symbol] = 0
                return 'sell'
        elif (z < -self.exit_threshold) or (np.abs(z) < self.exit_threshold):
            if state == -1:
                self.states[symbol] = 0
                return 'cover'
        #hold otherwise
        return 'hold'
