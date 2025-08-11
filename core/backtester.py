from execution.executor import Sizing
from execution.risk import RiskManager
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
import os
import statsmodels.api as sm

class Backtester:
    def __init__(self, symbols, strategy1, strategy2, executor, portfolio, data, models, stop_loss=0.10):
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        self.executor = executor
        self.portfolio = portfolio
        self.initial = self.portfolio.cash
        self.data = data
        self.stop_loss_threshold = stop_loss
        self.sizing = Sizing()
        self.portfolio_tracker = []
        self.portfolio_df = pd.DataFrame()
        self.stop_losses = {}
        self.prices = {}
        self.trade_df = pd.DataFrame
        self.models = models
        self.regime_counters = {symbol: 0.0 for symbol in symbols}

    def run(self):
        # i is the timestamp
        for i, group in self.data.groupby(self.data.index):
            for idx, row in group.iterrows():
                signal = self.strategy1.generate_signals(row)
                price = row['price']
                bid = row['bid_price']
                ask = row['ask_price']
                symbol = row['symbol']
                ret = row['return']

                regime = 0
                if self.models != None:
                    risk_model = self.models[symbol]
                    regime, probs = risk_model.update(ret)
                
                # check for stop-loss threshold
                stop_loss_signal = self.stop_loss(symbol, price)
                if stop_loss_signal is not None:
                    signal = stop_loss_signal
                    #print(f"Stop loss enabled at {i} for {symbol} @ {price}")
                    self.stop_losses[i] = {'symbol': symbol, 'price': price, 'signal': signal}

                #new regime
                if regime == 1:
                    print(f"Regime change at {i} for {symbol}")
                    if self.strategy2 != None:
                        if self.regime_counters[symbol] == 0:
                            self.executor.exit(self.portfolio, symbol, price, bid, ask, i)
                        signal = self.strategy2.generate_signals(row)
                    else:
                        signal = self.regime_change(symbol)
                    self.regime_counters[symbol] += 1
                else: 
                    if self.regime_counters[symbol] > 0:
                        self.executor.exit(self.portfolio, symbol, price, bid, ask, i)
                    self.regime_counters[symbol] = 0
                    

                # Determine position sizing
                if signal in ('buy', 'short'):
                    #no margin positions
                    if self.portfolio.cash > 0:
                        quantity = self.sizing.kelly_criterion(self.portfolio.current_value(), price)
                    else: quantity = 0
                else: 
                    quantity = self.portfolio.positions[symbol]['quantity']
                    if (quantity < 0 and signal == 'sell') or (quantity > 0 and signal == 'cover'):
                        quantity = 0
                    quantity = np.abs(quantity)

                self.executor.execute_trade(self.portfolio, symbol, signal, price, bid, ask, quantity, i)
                self.prices[f"{symbol}_price"] = price
            # keep record for evaluation later
            portfolio_value = self.portfolio.current_value()
            positions = {f"{symbol}_pos": (pos['quantity'] * pos['avg_price']) for symbol, pos in self.portfolio.positions.items()}
            unrealized_pnl = {f"{symbol}_unrealized_pnl": pos['unrealized_pnl'] for symbol, pos in self.portfolio.positions.items()}
            realized_pnl = {f"{symbol}_realized_pnl": pos['realized_pnl'] for symbol, pos in self.portfolio.positions.items()}
            record = {
                'timestamp': i,
                'value': portfolio_value,
                **self.prices,
                **positions,  # merge in by unpacking
                **unrealized_pnl,
                **realized_pnl,
            }

            self.portfolio_tracker.append(record)
        self.portfolio_df = pd.DataFrame(self.portfolio_tracker).set_index('timestamp')
        self.trade_df = pd.DataFrame(self.portfolio.trade_log)

    def stop_loss(self, symbol, current_price):
        """
        Check if stop loss threshold is met for a symbol and its current price.
        If so, return the correponding exit action. If not, return None.
        """
        quantity = self.portfolio.positions[symbol]['quantity']
        avg_price = self.portfolio.positions[symbol]['avg_price']
        if quantity > 0:  # Long position
            if current_price <= avg_price * (1 - self.stop_loss_threshold):
                return 'sell'
        
        elif quantity < 0:  # Short position
            if current_price >= avg_price * (1 + self.stop_loss_threshold):
                return 'cover'

        return None
    
    def regime_change(self, symbol):
        """
        Exit positions and hold until regime is changed back.
        """
        quantity = self.portfolio.positions[symbol]['quantity']
        if quantity > 0:
            return 'sell'
        elif quantity < 0:
            return 'cover'
        return 'hold'

    def normalized_assets(self, price_cols):
        normalized_prices= pd.DataFrame()
        for col in price_cols:
            normalized = self.portfolio_df[col] / self.portfolio_df[col].iloc[0]
            normalized_prices[col] = normalized
        return normalized_prices
    
    def portfolio_visual(self):
        price_cols = [col for col in self.portfolio_df.columns if col.endswith('_price')]

        normalized_prices= self.normalized_assets(price_cols)

        equal_weight_value = normalized_prices.mean(axis=1)
        equal_weight_value_scaled = equal_weight_value * self.initial
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

        # --- Top plot: Portfolio Value ---
        color1 = 'tab:blue'
        color2 = 'tab:gray'
        ax1.set_ylabel('Portfolio Value', color=color1)
        ax1.plot(self.portfolio_df.index, self.portfolio_df['value'], color=color1, label='Portfolio Value')
        ax1.plot(self.portfolio_df.index, equal_weight_value_scaled, color=color2, label='Equal-Weight Buy & Hold')
        ax1.set_title('Portfolio Value vs Equal-Weighted Benchmark')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # --- Bottom plot: Standardized Asset Prices ---
        ax2.set_ylabel('Normalized Asset Prices')

        for col in normalized_prices.columns:
            ax2.plot(self.portfolio_df.index, normalized_prices[col], label=col)


        ax2.legend(loc='upper left')
        ax2.set_title('Normalized Asset Prices')
        ax2.grid(True)

        # Set common X-axis label
        plt.xlabel('Time')

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig(f"portfolio.png")
        plt.close()


    def position_visual(self):

        # --- Right plot: Positions Over Time ---
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

        position_cols = [col for col in self.portfolio_df.columns if col.endswith('_pos')]

        self.portfolio_df['total_position'] = self.portfolio_df[position_cols].sum(axis=1)

        ax1.plot(self.portfolio_df.index, self.portfolio_df['total_position'], label='Total Position Value', color='tab:purple')
        ax1.set_title('Total Position Value Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position Value')


        self.portfolio_df[position_cols].plot(ax=ax2)
        ax2.set_title('Position Value Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Value')
        ax2.legend(title='Symbol')
        ax2.grid(True)

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig(f"position.png")
        plt.close()


    def pnl_visual(self):

        realized_pnl_cols = [col for col in self.portfolio_df.columns if col.endswith('_realized_pnl')]
        unrealized_pnl_cols = [col for col in self.portfolio_df.columns if col.endswith('_unrealized_pnl')]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot Realized PnL
        self.portfolio_df[realized_pnl_cols].plot(ax=ax1)
        ax1.set_title('Realized PnL Over Time')
        ax1.set_ylabel('PnL ($)')
        ax1.grid(True)
        ax1.legend(title='Symbol (Realized)')

        # Plot Unrealized PnL
        self.portfolio_df[unrealized_pnl_cols].plot(ax=ax2)
        ax2.set_title('Unrealized PnL Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('PnL ($)')
        ax2.grid(True)
        ax2.legend(title='Symbol (Unrealized)')


        plt.tight_layout()
        plt.savefig(f"pnl.png")
        plt.close()

        
    def overlays_visual(self):
        # Extract all unique asset symbols from the trade log
        log_df = pd.DataFrame(self.portfolio.trade_log)
        unique_symbols = log_df['symbol'].unique()
        for filename in os.listdir('visuals'):
            file_path = os.path.join('visuals', filename)
            if os.path.isfile(file_path) and file_path != "visuals\.gitkeep":
                os.remove(file_path)

        for symbol in unique_symbols:
            price_col = f"{symbol}_price"
            if price_col not in self.portfolio_df.columns:
                print(f"Skipping {symbol} — no price data found.")
                continue

            # Filter trade log for this symbol
            trades = log_df[log_df['symbol'] == symbol]
            trades = trades[trades['quantity'] != 0]

            buys = trades[trades['action'] == 'buy']
            sells = trades[trades['action'] == 'sell']
            shorts = trades[trades['action'] == 'short']
            covers = trades[trades['action'] == 'cover']


            # Plot price
            fig = plt.figure(figsize=(12, 6))
            plt.plot(self.portfolio_df.index, self.portfolio_df[price_col], label=f'{symbol} Price', color='blue')

            # Overlay trades
            plt.scatter(buys['timestamp'], buys['price'], color='green', marker='^', label='Buy', zorder=5)
            plt.scatter(sells['timestamp'], sells['price'], color='red', marker='v', label='Sell', zorder=5)
            plt.scatter(shorts['timestamp'], shorts['price'], color='orange', marker='v', label='Short', zorder=5)
            plt.scatter(covers['timestamp'], covers['price'], color='purple', marker='^', label='Cover', zorder=5)

            plt.title(f'{symbol} Price with Trade Entries/Exits')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            os.makedirs("visuals", exist_ok=True)
            plt.savefig(f"visuals/{symbol}.png")
            print(f"{symbol} plot saved to visuals.")
            plt.close(fig)

    def get_return(self):
        start_value = self.initial
        end_value = self.portfolio_df.iloc[-1]['value']
        return (end_value - start_value) / start_value
    
    def get_trade_counts(self):
        actions = [trade['action'] for trade in self.portfolio.trade_log]
        action_counts = Counter(actions)
        return action_counts

    def summary(self):
        """
        Provide some summary statistics and results for analysis.
        Returns:
        dict: All computed statistics for use in dashboards or reports.
        """
        
        def fmt_currency(x):
            return f"${x:,.2f}"

        def fmt_percent(x):
            return f"{x * 100:.2f}%"
        
        price_cols = [col for col in self.portfolio_df.columns if col.endswith('_price')]
        position_cols = [col for col in self.portfolio_df.columns if col.endswith('_pos')]

        unique_dates = self.portfolio_df.index.normalize().unique().sort_values()
        freq = len(self.portfolio_df.loc[unique_dates[-2]:unique_dates[-1]])
        
        start_value = self.initial
        end_value = self.portfolio_df.iloc[-1]['value']
        total_return = (end_value - start_value) / start_value
        time_range = self.portfolio_df.index[-1] - self.portfolio_df.index[0]
        years = time_range.total_seconds() / (365.25 * 24 * 60 * 60)
        cagr = ((end_value / start_value)**(1/years) - 1)

        portfolio_exposures = self.portfolio_df[position_cols].abs().sum(axis=1) / self.portfolio_df['value']
        ave_exposure = portfolio_exposures.mean()
        portfolio_returns = self.portfolio_df['value'].pct_change().dropna()
        portfolio_volatility = portfolio_returns.std()
        annualized_vol = portfolio_volatility * ((252 * freq)**0.5) 
        sharpe_ratio = cagr / annualized_vol
        normalized_returns = portfolio_returns / ave_exposure
        cumulative_return = (1 + normalized_returns).prod() - 1

        normalized_prices = self.normalized_assets(price_cols)
        mean_prices = normalized_prices.mean(axis=1) * self.initial
        start = mean_prices.iloc[0]
        end = mean_prices.iloc[-1]
        total_return_assets = (end - start) / start
        cagr_assets = ((end / start)**(1/years) - 1)
        asset_returns = mean_prices.pct_change().dropna()
        asset_volatility = asset_returns.std()
        annualized_asset_vol = asset_volatility * ((252 * freq)**0.5) 

        trades = self.trade_df[self.trade_df['realized_pnl'] != 0]
        win_rate = (len(trades[trades['realized_pnl'] > 0]) / len(trades))
        ave_return = trades['return'].mean()

        in_market = self.portfolio_df[position_cols].abs().sum(axis=1) > 0
        exposure_ratio = in_market.sum() / len(self.portfolio_df)

        X = asset_returns
        y = normalized_returns
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        alpha = model.params['const']
        beta = model.params[0]

        running_max = self.portfolio_df['value'].cummax()
        drawdowns = (self.portfolio_df['value'] - running_max) / running_max
        drawdowns = drawdowns.fillna(0)
        max_drawdown = drawdowns.min()
        in_drawdown = drawdowns < 0
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_periods = drawdown_periods * in_drawdown
        min_drawdowns = drawdowns.groupby(drawdown_periods).min()
        avg_drawdown = min_drawdowns.mean()

        actions = [trade['action'] for trade in self.portfolio.trade_log]
        action_counts = Counter(actions)

        # Store results in dictionary
        
        return {
            "Portfolio Summary": {
                "Start Value": fmt_currency(start_value),
                "End Value": fmt_currency(end_value),
                "Total Return": fmt_percent(total_return),
                "CAGR": fmt_percent(cagr),
                "Annualized Volatility": fmt_percent(annualized_vol),
                "Exposure-Adjusted Return": fmt_percent(cumulative_return)
            },
            "Benchmark Summary": {
                "Start Value": fmt_currency(start),
                "End Value": fmt_currency(end),
                "Total Return": fmt_percent(total_return_assets),
                "CAGR": fmt_percent(cagr_assets),
                "Annualized Volatility": fmt_percent(annualized_asset_vol)
            },
            "Results": {
                "Sharpe Ratio": round(sharpe_ratio, 3),
                "Win Rate": fmt_percent(win_rate),
                "Average Trade Return": fmt_percent(ave_return),
                "Average Exposure": fmt_percent(ave_exposure),
                "Exposure Time": fmt_percent(exposure_ratio),
                "Maximum Drawdown": fmt_percent(max_drawdown),
                "Average Drawdown": fmt_percent(avg_drawdown),
                "Alpha": f"{alpha:.8f}",
                "Beta": round(beta, 4)
            },
            "Trade Frequency": {
                k.capitalize(): v for k, v in action_counts.items()
            }
        }