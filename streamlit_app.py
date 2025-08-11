import streamlit as st
from core.engine import Engine
from strategies.algos import MeanReversion, MACD
from data.data import get_historical_bars, get_historical_quotes
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import product
import os
from execution.risk import RiskManager
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def param_heat_map(strategy_class, symbols, param1, param1_values, param2, param2_values):
    """
    Given two parameters, generate a heat map of portfolio returns to find the best set of parameters for the strategy.
    """
    results = pd.DataFrame(index=param1_values, columns=param2_values)
    window = 150
    end_date = "2025-08-04"
    start_date = "2024-01-10"
    start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=window)).strftime("%Y-%m-%d")
    data = get_historical_bars(*symbols, start_date=start_date, end_date=end_date)
    data['price'] = data['close']
    data['bid_price'] = data['price'] - 0.02
    data['ask_price'] = data['price'] + 0.02
    start_date = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"), tz="UTC")
    # Loop through parameter combinations
    for p1, p2 in product(param1_values, param2_values):
        #have to change the params for the strategy
        strategy = strategy_class(symbols=symbols, enter_threshold=p1, window=p2)
        # Run your backtest function here
        processed_data = strategy.data_process(data=data)
        processed_data = processed_data[processed_data.index > start_date]
        engine = Engine(symbols=symbols, data=processed_data, strategy=strategy)
        engine.run()
        portfolio_return = engine.portfolio_return
        # Store the result in the dataframe
        results.loc[p1, p2] = portfolio_return
        print(f"Portfolio Backtest for {param1}: {p1} and {param2}: {p2} completed.")
    print("Backtests completed, heat map rendering...")
    results = results.astype(float)
    plt.figure(figsize=(8, 6))
    sns.heatmap(results, annot=True, fmt=".2%", cmap="YlGnBu")
    plt.title("Portfolio Return Heatmap")
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.show()


st.set_page_config(layout="wide", page_title="Trading Dashboard", page_icon="📈")

st.title("Trading Engine Dashboard")

st.sidebar.title("Backtest")
strategy_name = st.sidebar.selectbox("Strategy", ["MACD/RSI Momentum", "Z-Score Mean Reversion", "Momentum + Mean Reversion Combo"])

st.header("Introduction")
st.write(""" This project features a trading engine simulator that takes a given strategy and studies its performance through a backtester, portfolio, executor, and risk
         manager I developed. I've also put together a few different trading strategies to test the engine. The backtester runs the strategy through 
         historical equity data and outputs critical performance metrics and visuals. These outputs are derived from
         the portfolio that tracks the strategy positions over the period, and the executor that makes trades with implemented transaction fees, bid/ask spread, and slippage. 
         Kelly Criterion is used for portfolio allocation by the executor once a signal is passed from the strategy to the backtester. The special addition to this engine
         is the risk manager, which is a Hidden Markov Model (HMM) trained on historical equity data before the backtest period to predict regime changes. When the 
         HMM predicts a regime change, any new positions are halted to minimize risk due to higher volatility. Since mean reversion can sometimes take advantage of 
         a high volatility market (with no drift), I've implemented a third strategy that substitutes the momentum strategy for a mean reversion strategy during a regime change.
""")

st.header("Instructions")
st.write("""The sidebar can be used to toggle the backtest parameters and strategy choice. Select 'On' for the risk manager to use the HMM in the backtesting. Since 
         the combination strategy only runs with the HMM for regime changes, the risk manager will automatically be implemented when 'Momentum + Mean Reversion Combo' is selected.
         Please enter tickers below that you would like the backtester to run the strategy through.
         The 'Run Backtest' button will collect Alpaca OHLC data from the time range you've specified, run it through the backtester with the given strategy,
         and then produce a grouping of visuals and metrics to better understand the performance of the strategy.
""")
example_tickers = ["SPY", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

st.markdown(
    f"**Example tickers:** {', '.join(example_tickers)}"
)

# Text input with default tickers
user_input = st.text_input(
    "Enter tickers (comma-separated):",
    value=", ".join(example_tickers),
    help="Enter stock symbols separated by commas (e.g., AAPL, TSLA, AMZN)"
)

# Convert to list and strip spaces
symbols = [ticker.strip().upper() for ticker in user_input.split(",") if ticker.strip()]
st.write("**Tickers to use:**", symbols)
initial_capital = st.sidebar.number_input("Initial capital", value=100000, step=1000)
transaction_cost = st.sidebar.number_input("Transaction cost (fraction)", value=0.001, format="%.6f")
start_date = st.sidebar.text_input("Start date of backtest", value='2024-07-01')
feedback_placeholder = st.sidebar.empty()
error = 0
valid_date = None
if start_date:
    try:
        valid_date = datetime.strptime(start_date, "%Y-%m-%d").date()

        if valid_date > datetime.today().date():
            feedback_placeholder.error("Start dates cannot be in the future. Please try again.")
            error +=1
        else: feedback_placeholder.success(f"Valid date: {valid_date}")
    except ValueError:
        feedback_placeholder.error("Invalid date format. Please enter date as YYYY-MM-DD.")
        error +=1
else:
    feedback_placeholder.error("Need to enter a valid start date.")
    error +=1
end_date = st.sidebar.text_input("End date of backtest", value=datetime.today().strftime('%Y-%m-%d'))
feedback_placeholder2 = st.sidebar.empty()
if end_date:
    try:
        valid_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        if valid_date > datetime.today().date():
            feedback_placeholder2.error("End dates cannot be in the future. Please try again.")
            error +=1
        else:
            if valid_date < datetime.strptime(start_date, "%Y-%m-%d").date():
                feedback_placeholder2.error("End dates cannot be before start date. Please try again.")
                error +=1
            else: feedback_placeholder2.success(f"Valid date: {valid_date}")
    except ValueError:
        feedback_placeholder2.error("Invalid date format. Please enter date as YYYY-MM-DD.")
        error +=1
else:
    feedback_placeholder2.error("Need to enter a valid end date.")
    error +=1

force_risk_on = (strategy_name == "Momentum + Mean Reversion Combo")

if force_risk_on:
    # Force 'On' and disable changing
    risk = st.sidebar.selectbox("Risk Manager", ['On', 'Off'], index=0, disabled=True)
else:
    # Normal selectable
    default_index = 0 if 'risk' not in st.session_state else (0 if st.session_state.risk == 'On' else 1)
    risk = st.sidebar.selectbox("Risk Manager", ['On', 'Off'], index=default_index)

run_button = st.sidebar.button("Run backtest")

if run_button and error == 0:
    #give enough data for processing to give rolling values
    train_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=75)).strftime("%Y-%m-%d")
    if risk == 'On':
        train_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=10*252)).strftime("%Y-%m-%d")

    data = get_historical_bars(*symbols, start_date=train_date, end_date=end_date)
    data['price'] = data['close']
    #simulating bid and ask since this is bar data
    data['bid_price'] = data['price'] - 0.02
    data['ask_price'] = data['price'] + 0.02

    strategy = None
    strategy2 = None
    finished_data = None
    if strategy_name == "MACD/RSI Momentum":
        strategy = MACD(symbols=symbols)
        finished_data = strategy.data_process(data=data)
    elif strategy_name == "Momentum + Mean Reversion Combo":
        strategy = MACD(symbols=symbols)
        strategy2 = MeanReversion(symbols=symbols)
        processed_data = strategy.data_process(data=data)
        finished_data = strategy2.data_process(data=processed_data)
    else:
        strategy = MeanReversion(symbols=symbols)
        finished_data = strategy.data_process(data=data)

    start_date = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"), tz="UTC")
    train_data = finished_data[finished_data.index < start_date]
    test_data = finished_data[finished_data.index >= start_date]

    st.write("Data Processed...")

    #clear model folder before starting
    for filename in os.listdir('models'):
        file_path = os.path.join('models', filename)
        if os.path.isfile(file_path) and file_path != "models/.gitkeep":
            os.remove(file_path)
    models = None
    if risk == 'On':
        models = {}
        for symbol, group in train_data.groupby(train_data['symbol']):
            st.write(f"{symbol} Model training...")
            risk = RiskManager(model_path=f"models/{symbol}_model")
            risk.train(group['return'])
            models[symbol] = risk

    st.write("Engine Running...")
    engine = Engine(symbols=symbols, data=test_data, strategy1=strategy, strategy2=strategy2, risk_models=models, initial_cash=initial_capital, transaction_cost=transaction_cost)
    engine.run()

    st.header("Results")
    st.write("""Metrics relevant to the strategy performance are outputted below. The portfolio return is normalized by dividing its returns by the average exposure
             so we can see the returns with 100% portfolio allocation. Since the portfolio is not fully allocated at all times, the annualized volatility for 
             the portfolio and benchmark do not make sense to compare. For a detailed comparison of the strategy portfolio to its equal-weight buy and hold benchmark,
             review the alpha and beta metrics. The hope is that when the risk manager is being implemented, the maximum and average drawdowns are decreased.""")
    results = engine.summary
    
    num_sections = len(results)
    cols = st.columns(num_sections)  # One column per section

    for col, (section, metrics) in zip(cols, results.items()):
        with col:
            st.subheader(section)
            for metric, value in metrics.items():
                st.markdown(f"**{metric}:** {value}")

    st.header("Visuals")

    st.write("""
        Below is a plot of the strategy portfolio value over time against a benchmark portfolio starting with the same initial value and investing equally in the given assets for the
             entire duration of the backtest (Buy and hold). Keep in mind that the strategy uses kelly criterion for position sizing, meaning it usually does not
             have full allocation of the portfolio throughout the backtest period. For this reason, the strategy portfolio may appear to have a smaller return and 
             volatility than the benchmark portfolio, although this may be untrue if we normalize the strategy portfolio.
             Also pictured are the normalized asset prices for reference.
""")
    st.image("portfolio.png", caption="Portfolio vs Equal-Weighted Benchmark", use_container_width=True)

    st.write(""" Both the realized and unrealized profit and loss for the portfolio's assets are pictured below. Through the unrealized pnl lense we can better
             see what opportunities our strategy is picking up and what troughs are slowing it down. 
""")
    st.image("pnl.png", caption="Profit & Loss over time", use_container_width=True)

    st.write(""" The total position value of the portfolio and the indivdual position values in the assets can both be viewed below. The portfolio position value
             is easier to gain insights from when there are many assets in the portfolio that fill the lower chart. For one to a couple assets, the asset position value
             visual is a better gauge to see what the strategy is investing in and for how much.
""")
    st.image("position.png", caption="Portfolio and Asset Positions over time", use_container_width=True)

    st.write("Trade logs are recorded and overlayed onto asset prices to see where the strategy is having success and with what positions.")
    for i in range(0, len(symbols), 2):
        row_images = symbols[i:i+2]
        cols = st.columns(len(row_images))
        for col, symbol in zip(cols, row_images):
            col.image(f"visuals/{symbol}.png", caption=symbol, use_container_width=True)

    log_df = pd.read_csv("logs.csv")

    st.write("""The realized pnl takes the current price and quantity we are exiting the position with and compares it to the average price of the positions with the same quantity.
             The returns row is based on that trades profit over the asset's position value, not the portfoltio value. The price of the trade in the logs is based on the current
             price of the asset - the actual price of the asset has transaction fees, bid-ask spread, and slippage added to it.
             """)
    st.markdown(
    f"""
    <div style="height:300px; overflow:auto; border:1px solid #ccc; padding:5px;">
        {log_df.to_html(index=False)}
    </div>
    """,
    unsafe_allow_html=True
)
    