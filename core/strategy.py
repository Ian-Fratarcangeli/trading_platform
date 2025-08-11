from abc import ABC, abstractmethod

class StrategyBase(ABC):
    def __init__(self, symbol):
        self.symbol = symbol

    @abstractmethod
    def generate_signals(self, point):
        """
        Returns a pandas Series of signals:
        1 = long, -1 = short, 0 = flat
        """
        pass

    @abstractmethod
    def data_process(self, data):
        """
        Processes the data to be used by the strategy.
        Returns a pandas DataFrame with the processed data.
        """
        pass

    @abstractmethod
    def data_visual(self, data):
        """
        Visualizes the processed data.
        """
        pass
