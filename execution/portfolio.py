import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, symbols, initial_cash):
        self.cash = initial_cash
        self.positions = {  
            symbol: {
            'quantity': 0.0,
            'avg_price': 0.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
        } for symbol in symbols}
        self.prices = {symbol: 0.0 for symbol in symbols}
        self.trade_log = []

    def get_position(self, symbol):
        return self.positions.get(symbol, 0)
    
    def update_prices(self, symbol, price):
        self.prices[symbol] = price

    def update_position(self, symbol, quantity, trade_price, action):
        """
        Modifies the cash and position for a trade.
        action: (str) buy, sell, short, or cover
        """
        realized = 0
        returns = 0
        # take out cash and add to position for buying or covering
        if action in ('buy', 'cover'):
            self.cash -= quantity * trade_price
            pos = self.positions[symbol]
            if action == 'buy':
                total_pos = np.abs(pos['quantity']) * pos['avg_price']
                new_total_pos = total_pos + (quantity * trade_price)
                new_quantity = np.abs(pos['quantity']) + quantity

                pos['avg_price'] = new_total_pos / new_quantity
                pos['quantity'] = new_quantity

            if action == 'cover':
                realized = (pos['avg_price'] - trade_price) * quantity
                returns = realized / (pos['avg_price'] * quantity)
                pos['realized_pnl'] += realized
                pos['quantity'] += quantity
                if pos['quantity'] == 0:
                    pos['avg_price'] = 0

            self.positions[symbol] = pos

        else:  # sell or short
            self.cash += quantity * trade_price
            pos = self.positions[symbol]
            if action == 'sell':
                realized = (trade_price - pos['avg_price']) * quantity
                returns = realized / (pos['avg_price'] * quantity)
                pos['realized_pnl'] += realized
                pos['quantity'] -= quantity
                if pos['quantity'] == 0:
                    pos['avg_price'] = 0
            
            if action == 'short':
                total_pos = np.abs(pos['quantity']) * pos['avg_price']
                new_total_pos = total_pos + (quantity * trade_price)
                new_quantity = np.abs(pos['quantity']) + quantity

                pos['avg_price'] = new_total_pos / new_quantity
                pos['quantity'] -= quantity

            self.positions[symbol] = pos
        return realized, returns

    def current_value(self):
        """
        Revalue all positions using current prices.
        """
        value = self.cash
        for symbol, data in self.positions.items():
            current_price = self.prices.get(symbol, 0)
            value += data['quantity'] * current_price
            #update the realized pnl in here
            data['unrealized_pnl'] = (current_price - data['avg_price']) * data['quantity']
            self.positions[symbol] = data
        return value

    def record_trade(self, symbol, quantity, current_price, action, pnl, returns, timestamp):
        """
        Log the trade.
        """

        trade = {
        'timestamp': timestamp,
        'symbol': symbol,
        'quantity': quantity,
        'price': current_price,
        'action': action,
        'realized_pnl': pnl,
        'return': returns
        }
        self.trade_log.append(trade)

    def view_logs(self):
        """
        Give a pretty print version of the trade logs.
        """
        print("-" * 80)
        print(f"{'Time':<20} {'Symbol':<8} {'Action':<8} {'Qty':<6} {'Price':<10} {'PnL':<10} {'Return':<6}")
        print("-" * 80)
        for trade in self.trade_log:
            str_time = trade['timestamp'].strftime("%Y-%m-%d %H:%M")
            print(f"{str_time:<20} {trade['symbol']:<8} {trade['action']:<8} "
                f"{trade['quantity']:<6.2f} ${trade['price']:<10.2f} {trade['realized_pnl']:<12.2f} {trade['return']:<8.2%}")
        print("-" * 80)
    def save_logs(self, path):
        """
        Save the trade logs to a given file path.
        """
        df = pd.DataFrame(self.trade_log)
        df.to_csv(path, index=False)
