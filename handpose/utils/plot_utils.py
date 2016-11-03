import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_imp_x(imps, frame_index, frame_shift, class_name, 
               subjects=None):
    """
    Plot impedance in spatial space.

    Parameters
    ----------
    imps: list
        List of impedance arrays.
    frame_index: int
        Frame index.
    frame_shift: int
        Shift frames.
    class_name: str
        Class name.
    subjects: list
        List of subject's name.

    """  
    num_imps = len(imps)

    if subjects == None:
        subjects = []
        for i in range(num_imps):
             subjects.append("subject{}".format(i+1))

    rows, cols = imps[0].shape
    x = range(cols)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

    for i in range(num_imps):
        axes[i].plot(x, imps[i][frame_index], x, imps[i][frame_index+frame_shift])
        axes[i].set_title('class: {}, {}'.format(class_name, subjects[i]))
        axes[i].set_xlabel('electrod index')
        axes[i].set_ylabel('impedances')
    
    return fig, axes

def plot_imp_t(df, tx=0, rx=1):
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


