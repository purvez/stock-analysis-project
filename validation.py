import pandas as pd
from analysis import compute_sma

def validate_sma():
    test_prices = [10, 20, 30, 40, 50]
    sma_manual = [None, None, 20.0, 30.0, 40.0]  # window=3
    sma_computed = compute_sma(test_prices, 3)
    return sma_manual == sma_computed
