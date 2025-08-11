from core.backtester import Backtester
from execution.portfolio import Portfolio
from execution.executor import Executor

class Engine:
    def __init__(self, symbols, data, strategy1, strategy2=None, risk_models=None, initial_cash=100000, transaction_cost=0.001):
        self.symbols = symbols
        self.data = data
        self.portfolio_return = 0
        self.counts = {}
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        self.models = risk_models
        self.cash = initial_cash
        self.transaction_cost = transaction_cost
        self.summary = None

    def run(self):
        print("-------Engine Running--------")
        executor = Executor(transaction_cost=self.transaction_cost)
        portfolio = Portfolio(self.symbols, initial_cash=self.cash)
        backtester = Backtester(self.symbols, self.strategy1, self.strategy2, executor, portfolio, self.data, self.models)
        print("-------Backtester Running--------")
        backtester.run()
        print("-------Backtester Complete--------")
        backtester.portfolio_visual()
        backtester.overlays_visual()
        backtester.position_visual()
        backtester.pnl_visual()
        portfolio.save_logs("logs.csv")
        self.portfolio_return = backtester.get_return()
        self.counts = backtester.get_trade_counts()
        self.summary = backtester.summary()

    