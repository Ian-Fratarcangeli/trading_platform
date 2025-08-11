import numpy as np

class Executor:
    def __init__(self, transaction_cost=0.001, slippage=0.0):
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def execute_trade(self, portfolio, symbol, signal, price, bid, ask, quantity, timestamp):
        """
        Executes a trade given a signal - buy, sell, short, cover, or hold.
        """
        portfolio.update_prices(symbol, price)
        if quantity != 0 and signal != 'hold':
            #account for transaction costs and spread
            buy_price = ask * (1 + self.transaction_cost)
            sell_price = bid * (1 - self.transaction_cost)
            profit = 0

            #get assets at the price with transaction cost
            if signal in ('buy', 'cover'):
                profit, returns = portfolio.update_position(symbol=symbol, quantity=quantity, trade_price = buy_price, action=signal)

            #sell assets at price minus transaction cost
            if signal in ('sell', 'short'):
                profit, returns = portfolio.update_position(symbol=symbol, quantity=quantity, trade_price = sell_price, action=signal)

            portfolio.record_trade(symbol=symbol, quantity=quantity, current_price=price, action=signal, pnl=profit, returns=returns, timestamp=timestamp)

    def exit(self, portfolio, symbol, price, bid, ask, timestamp):
        """
        Exit all positions in the given asset
        """
        quantity = portfolio.positions[symbol]['quantity']
        if quantity > 0:
            signal = 'sell'
            self.execute_trade(portfolio, symbol, signal, price, bid, ask, quantity, timestamp)
        if quantity < 0:
            signal = 'cover'
            self.execute_trade(portfolio, symbol, signal, price, bid, ask, np.abs(quantity), timestamp)
        
    def calculate_trade_price(self, price, signal):
        """
        Apply slippage if needed.
        """
        pass

class Sizing:
    def __init__(self):
        pass

    def fixed_dollar(self, price, dollars=10000):
        """
        Determines the quantity of shares to trade by a fixed dollar amount.
        """
        return dollars / price
        
    
    def fixed_percent(self, portfolio_value, price, percent=0.1):
        """
        Determines the quantity of shares to trade by a fixed percentage of the portfolio.
        """
        return (portfolio_value * percent) / price
    
    def kelly_criterion(self, portfolio_value, price, win_rate=0.6, risk_reward = 1):
        """
        Determines the quantity of shares to trade by the kelly criterion.
        The parameters have been validated through backtesting.
        """
        kelly = ((risk_reward * win_rate) - (1 - win_rate)) / risk_reward
        return (portfolio_value * kelly) / price
    