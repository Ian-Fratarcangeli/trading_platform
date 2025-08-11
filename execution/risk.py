import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator
from datetime import datetime

class RiskManager:
    def __init__(self, model_path=None, n_states=2, window=20, retrain=False):
        self.model_path = model_path
        self.n_states = n_states
        self.model = None
        self.window = window
        self.buffer = []
        if model_path and os.path.exists(model_path) and not retrain:
            self.load_model()
        else:
            self.model = None

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def save_model(self):
        if self.model_path:
            joblib.dump(self.model, self.model_path)

    def train(self, returns):
        """
        Train HMM on features. feature_df should be DataFrame of features (rows = time, columns = features).
        """
        X = returns.values.reshape(-1, 1)
        model = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=1000, random_state=42)
        model.fit(X)
        self.model = model
        self.save_model()
        self.buffer = returns.values[-self.window:].tolist()
    

    def update(self, new_return):
        self.buffer.append(new_return)

        if len(self.buffer) > self.window:
            self.buffer.pop(0)

        if len(self.buffer) < self.window:
            # Not enough data yet; return a default or None
            return None, None
        data = np.array(self.buffer).reshape(-1, 1)
        states = self.model.predict(data)
        probs = self.model.predict_proba(data)

        # Return last timestep state and probabilities
        return states[-1], probs[-1]

    
    def plot_in_sample_hidden_states(self, states, probs, df):
        """
        Plot the adjusted closing prices masked by 
        the in-sample hidden states as a mechanism
        to understand the market regimes.
        """

        state_probs_df = pd.DataFrame(
        probs,
        index=df.index,  # your time index
        columns=[f"State {i}" for i in range(self.n_states)]
        )

        # Plot stacked probabilities over time
        fig, ax = plt.subplots(figsize=(14, 6))
        state_probs_df.plot.area(ax=ax, alpha=0.6)
        ax.set_title("HMM State Probabilities Over Time")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Time")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        # Create the correctly formatted plot
        fig, axs = plt.subplots(
            self.n_states, 
            sharex=True, sharey=True
        )
        colours = cm.rainbow(
            np.linspace(0, 1, self.n_states)
        )
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            mask = states == i
            ax.plot(
                df.index[mask], 
                df["price"][mask], 
                ".", linestyle='none', 
                c=colour
            )
            ax.set_title("Hidden State #%s" % i)
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.grid(True)
        plt.show()