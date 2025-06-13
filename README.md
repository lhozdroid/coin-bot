# Coin Bot

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-ready-green.svg)]()

Coin Bot is a deep learning pipeline for training and evaluating the **Squeeze Momentum Transformer** model on financial time series data. It is designed for efficient pattern recognition in candle chart data using PyTorch.

## Overview

This project includes:

- A PyTorch dataset for candlestick data
- The Squeeze Momentum Transformer (SMT) model
- A complete training pipeline with validation and testing
- CUDA support for GPU acceleration
- Configurable settings using YAML

## Project Structure

```

coin-bot/
├── main_train.py                        # Main entry point to train the model
├── cuda_test.py                         # GPU and CUDA diagnostic
├── data/                                # Preprocessed .npy files
├── model/
│   ├── candle_dataset.py                # Dataset utilities
│   ├── squeeze_momentum_transformer.py  # SMT model definition
│   ├── training.py                      # Training pipeline
│   └── config.yml                       # Model/training config
├── optimize/                            # Hyperparameter tuning
├── environment.yml                      # Conda environment definition
└── package.sh                           # Packaging utility

````

## Setup

### Clone the repository

```bash
git clone https://github.com/lhozdroid/coin-bot.git
cd coin-bot
````

### Environment setup (Conda)

```bash
conda env create -f environment.yml
conda activate coin-bot
```

## GPU Support

Run the following to test if CUDA is available:

```bash
python cuda_test.py
```

Expected output:

```
CUDA Available: True
GPU Name: NVIDIA GeForce ...
```

## Training the Model

Ensure the following files are in the `data/` directory:

* `features.npy`: Input sequences
* `labels.npy`: Corresponding labels

Then run:

```bash
python main_train.py
```

Hyperparameters are configured in `model/config.yml`.

## Model Details

The Squeeze Momentum Transformer (SMT) is a transformer-based model tailored for financial time series. It features:

* Multi-head self-attention
* Lightweight design
* Positional embeddings
* Configurable dropout and depth

## Configuration Example

Example `model/config.yml`:

```yaml
input_size: 10
hidden_size: 64
num_heads: 4
num_layers: 3
dropout: 0.1
epochs: 20
learning_rate: 0.0005
```

## Future Improvements

* Backtesting module
* Live market inference
* Integration with financial APIs
* Experiment tracking tools (MLflow, WandB)

## License

MIT © [lhozdroid](https://github.com/lhozdroid)