from alpaca.data.live.stock import StockDataStream
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
import asyncio
import asyncpg
import os
from collections import deque
from dotenv import load_dotenv
from datetime import datetime
import keyboard
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")

# Set up the data stream and client using IEX
stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY) 
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


async def exit():
    print("Press Ctrl+q to quit the stream.")
    while True:
        if keyboard.is_pressed('ctrl+q'):
            break
        await asyncio.sleep(0.1)  # small pause to avoid CPU overuse

async def run_stream():
    try:
        await stream._run_forever()
    except asyncio.CancelledError:
        print("run_stream() was cancelled.")
    finally:
        try:
            await stream.close()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        print("WebSocket stream closed.")

def make_bar_handler(db):
    async def bar_handler(bar):
        await db.execute(
            '''INSERT INTO bars (symbol, timestamp, open, high, low, close, volume)
               VALUES ($1, $2, $3, $4, $5, $6, $7)''',
            bar.symbol, bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume
        )
        print(f"Inserted bar for {bar.symbol} @ {bar.timestamp}")
    return bar_handler

#this is for the data feed- run this with asyncio to stream
async def live_feed(*symbols):
    # Connect to database
    db = await asyncpg.connect(user='postgres', password=DATABASE_PASSWORD,
                                 database='trading', host='127.0.0.1')
    handler = make_bar_handler(db)
    # Subscribe to trade updates for a symbol (e.g. AAPL)
    stream.subscribe_bars(handler, *symbols)
    print("Streaming live quote data (IEX)...")
    
    stream_task = asyncio.create_task(run_stream())

    await exit()

    # Shutdown
    print("Cancelling stream task...")
    stream_task.cancel()
    await stream_task

def get_historical_bars(*symbols, start_date, end_date, timeframe='D'):
    """
    Get historical equity bars for symbol(s) and timeframe
    """
    time = TimeFrame.Day
    if timeframe == 'H':
        time = TimeFrame.Hour
    if timeframe == 'M':
        time = TimeFrame.Minute
        
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=time,
        start=start,
        end=end
    )

    data = client.get_stock_bars(request_params).df
    #split multicolumn index and make it just the time
    data = data.reset_index()
    data.set_index('timestamp', inplace=True)
    # Fetch data
    return data

def get_historical_quotes(*symbols, start_date, end_date, resample='1min' ):
    """
    Get historical equity quotes for symbol(s) and timeframe, with a resampling argument
    resample: (str)
        1s => 1 second
        5s => 5 seconds
        1min => 1 minute
        15min -> 15 minutes
        1H => 1 hour
        1D => 1 day
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    request_params = StockQuotesRequest(
        symbol_or_symbols=symbols,
        start=start,
        end=end
    )

    data = client.get_stock_quotes(request_params).df
    
    #split multicolumn index and make it just the time
    data = data.reset_index()
    data.set_index('timestamp', inplace=True)

    resampled_data = data.resample(resample).last().dropna()

    # Fetch data
    return resampled_data
