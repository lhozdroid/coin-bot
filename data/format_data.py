# Ensures that the script can be run from any working directory by setting up the root path
import os
import sys

# Compute the root path relative to this file's location
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path if not already present
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
from sklearn.preprocessing import StandardScaler


def load_config(path="data/config.yml"):
    """
    Loads the data format configuration.

    Args:
        path:

    Returns:

    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(path):
    """
    Loads the data from the CSV file

    Args:
        path:

    Returns:

    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def compute_indicators(df, config):
    """
    Computes all the technical indicators

    Args:
        df:
        config:

    Returns:

    """
    ta_windows = config["ta_windows"]
    squeeze_len = ta_windows["squeeze"]
    bb_std = ta_windows["bb_std"]
    kc_mult = ta_windows["kc_mult"]

    # Ensure BB and KC bands exist
    df.ta.bbands(length=squeeze_len, std=bb_std, append=True)
    df.ta.kc(length=squeeze_len, scalar=kc_mult, append=True)

    # Squeeze Momentum
    df.ta.squeeze(length=squeeze_len, bb_std=bb_std, kc_mult=kc_mult, append=True)
    original_name = f"SQZ_{squeeze_len}_{bb_std}_{squeeze_len}_{kc_mult}"
    df.rename(columns={original_name: "squeeze"}, inplace=True)

    # Other indicators
    df["momentum"] = ta.roc(df["close"], length=ta_windows["momentum"])
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ta_windows["atr"])
    df["roc"] = ta.roc(df["close"], length=ta_windows["roc"])
    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=ta_windows["adx"])["ADX_" + str(ta_windows["adx"])]

    df = df.set_index("timestamp")
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"], length=ta_windows["vwap"])
    df = df.reset_index()

    df["vol_mean"] = df["volume"].rolling(window=ta_windows["volume_z"]).mean()
    df["vol_std"] = df["volume"].rolling(window=ta_windows["volume_z"]).std()
    df["volume_z"] = (df["volume"] - df["vol_mean"]) / df["vol_std"]

    df["bb_width"] = df[f"BBU_{squeeze_len}_{bb_std}"] - df[f"BBL_{squeeze_len}_{bb_std}"]
    df["kc_width"] = df[f"KCUe_{squeeze_len}_{kc_mult}"] - df[f"KCLe_{squeeze_len}_{kc_mult}"]

    return df.drop(columns=["vol_mean", "vol_std"])


def get_feature_columns():
    """
    Obtains all the feature columns

    Returns:

    """
    return ["open", "high", "low", "close", "volume", "squeeze", "bb_width", "kc_width", "momentum", "atr", "roc", "adx", "vwap", "volume_z"]


def normalize_features(df, feature_cols):
    """
    Normalizes the values of the features

    Args:
        df:
        feature_cols:

    Returns:

    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


def label_from_future(current_close, future_close, up, down):
    """
    Assigns the labels to the training data

    Args:
        current_close:
        future_close:
        up:
        down:

    Returns:

    """
    delta = (future_close - current_close) / current_close
    if delta > up:
        return 0
    elif delta < down:
        return 1
    else:
        return 2


def generate_samples(df, feature_cols, config):
    """
    Using the window size it generates the data for training
    Args:
        df:
        feature_cols:
        config:

    Returns:

    """

    # Obtains the window size
    window_size = config["window_size"]

    # Obtains the prediction offset
    offset = config["prediction_offset"]

    # Obtains the threshold for up and down indicator
    up = config["thresholds"]["up"]
    down = config["thresholds"]["down"]

    # Generates all the features and labels
    features, labels = [], []
    for i in range(len(df) - window_size - offset):
        window = df.iloc[i:i + window_size][feature_cols].values
        current_close = df["close"].iloc[i + window_size - 1]
        future_close = df["close"].iloc[i + window_size + offset - 1]
        label = label_from_future(current_close, future_close, up, down)
        features.append(window)
        labels.append(label)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    """
    Start point of data formatting

    Returns:

    """
    print("Loading configuration...")
    config = load_config()

    print("Reading raw data...")
    df = load_data(config["input_file"])

    print("Computing indicators...")
    df = compute_indicators(df, config)

    print("Dropping NaNs and normalizing...")
    df = df.dropna().reset_index(drop=True)
    feature_cols = get_feature_columns()
    df = normalize_features(df, feature_cols)

    print("Generating training samples...")
    X, y = generate_samples(df, feature_cols, config)

    print(f"Saving {len(X)} samples...")
    np.save(f"data/{config['output_features']}", X)
    np.save(f"data/{config['output_labels']}", y)

    print("Data formatting complete.")


if __name__ == "__main__":
    main()
