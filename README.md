# HyperIV: Real-time Implied Volatility Smoothing

HyperIV constructs an arbitrage-free implied volatility surface from a few (e.g., 9) option contracts in 2 milliseconds.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) as the dependency manager. To install uv, please follow the instructions in their [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

Once uv is installed, run the following command to install all required dependencies:

```bash
uv sync
```

## Usage

The project includes two main Jupyter notebooks:

1. `data_prep.ipynb`: Demonstrates the data preparation pipeline for option market data
2. `model_train.ipynb`: Contains code for model training and evaluation