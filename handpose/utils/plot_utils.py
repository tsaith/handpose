import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_imp_hist(df, tx=0, rx=1):
    """
    Plot impedance values including real and imaginary parts.

    Parameters
    ----------
    df: DataFrame
        Data frame
    tx: int
        TX index
    rx: int
        RX index
    """  
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    
    col_name1 = "imp_tx_{}_rx_{}_real".format(tx, rx)
    col_name2 = "imp_tx_{}_rx_{}_real".format(rx, tx)
    
    df[[col_name1, col_name2]].plot(ax=axes[0],  logy=False)
    axes[0].set_xlabel('time index')
    col_name1 = "imp_tx_{}_rx_{}_img".format(tx, rx)
    col_name2 = "imp_tx_{}_rx_{}_img".format(rx, tx)
  
    df[[col_name1, col_name2]].plot(ax=axes[1])
    axes[1].set_xlabel('time index')

