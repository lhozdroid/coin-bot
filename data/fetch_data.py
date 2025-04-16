# Ensures that the script can be run from any working directory by setting up the root path
import os
import sys

# Compute the root path relative to this file's location
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path if not already present
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import ccxt
import pandas as pd


def fetch_ohlcv_chunk(exchange, symbol, timeframe, since_ms):
    """
    Fetches a chunk of OHLCV values

    Args:
        exchange:
        symbol:
        timeframe:
        since_ms:

    Returns:

    """
    try:
        # Obtains a total of 1000 candles starting at the indicated timeframe
        candles = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since_ms, limit=1000)
        time.sleep(1.2)
        return candles
    except Exception as e:
        print(f"[ERROR] Failed to fetch chunk starting at {datetime.utcfromtimestamp(since_ms / 1000)}: {e}")
        return []


def generate_chunk_timestamps(start_ms, end_ms, interval_minutes, chunk_size=1000):
    """
    Generates chunks of timestamps to later obtain the information

    Args:
        start_ms:
        end_ms:
        interval_minutes:
        chunk_size:

    Returns:

    """
    interval_ms = interval_minutes * 60 * 1000
    chunk_duration_ms = chunk_size * interval_ms
    return list(range(start_ms, end_ms, chunk_duration_ms))


def fetch_all_chunks_parallel(start_date, end_date, symbol='BTC/USD', timeframe='5m', max_threads=5):
    """
    Obtains all the chunks of timestamps in parallel

    Args:
        start_date:
        end_date:
        symbol:
        timeframe:
        max_threads:

    Returns:

    """

    # Coin base exchange endnpoint
    exchange = ccxt.coinbase({'enableRateLimit': True})

    # Minute interval
    interval_minutes = int(timeframe.replace('m', ''))

    # Start and end dates
    start_ms = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end_date}T00:00:00Z")

    # Generates the sections of timestamps
    timestamps = generate_chunk_timestamps(start_ms, end_ms, interval_minutes)

    # Creates a variable to save all the candles
    all_candles = []

    # Starts the processing of the timestamp chunks
    print(f"Starting fetch from {start_date} to {end_date} using {len(timestamps)} chunks...")
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Creates all the futures
        futures = {executor.submit(fetch_ohlcv_chunk, exchange, symbol, timeframe, ts): ts for ts in timestamps}

        # Awaits the futures to complete
        for future in as_completed(futures):
            # Obtains the timestamp
            timestamp = futures[future]
            try:
                # Obtains the candles
                candles = future.result()

                # Checks that there are candles
                if candles:
                    # Saves them
                    all_candles.extend(candles)
                    print(f"[OK] Chunk from {datetime.utcfromtimestamp(timestamp / 1000)}: {len(candles)} rows")
                else:
                    # Prints a warning
                    print(f"[WARN] Empty chunk from {datetime.utcfromtimestamp(timestamp / 1000)}")
            except Exception as e:
                print(f"[ERROR] Thread failed at {datetime.utcfromtimestamp(timestamp / 1000)}: {e}")

    print(f"Fetched {len(all_candles)} total rows.")
    return all_candles


def save_to_csv(candles, filename):
    """
    Saves the values into CSV file

    Args:
        candles:
        filename:

    Returns:

    """
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def main():
    """
    Starts processing

    Returns:

    """

    # Configuration of the script
    start_date = "2020-04-01"
    end_date = "2025-04-01"
    symbol = "BTC/USD"
    timeframe = "5m"
    output_file = "data.csv"

    # Obtains the data into memory
    candles = fetch_all_chunks_parallel(start_date=start_date, end_date=end_date, symbol=symbol, timeframe=timeframe, max_threads=5)

    # Saves the CSV file
    save_to_csv(candles, output_file)


if __name__ == "__main__":
    main()
