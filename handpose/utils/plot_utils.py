import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

  
def plot_imu_data(data, xlabel="xlabel", ylabel="ylabel", title="title", legend=["", "", ""]):
    """
    Plot the IMU data.

    Parameters
    ----------
    data: 2d-array
        Sensor data. 
        The rows are for time and columns are for directions. 
    xlabel: str
       Label of x-axis.
    ylabel: str
       Label of y-axis.
    title: str
       Title of figure.
    legend: list   
       Legend of data.
    """  
    rows, cols = data.shape
    if cols != 3:
        print("Error: there should be 3 columns in the input data.")

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))  
    x = range(rows)

    axes.plot(x, data[:, 0], x, data[:, 1], x, data[:, 2])
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend(legend)

    return fig, axes 


def plot_imp_x(imps, frame_index, frame_shift, class_name, 
               notes=None):
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
    notes: list
        List of notes written in the tilte.

    """  
    num_imps = len(imps)

    if notes == None:
        notes = []
        for i in range(num_imps):
             notes.append("subject{}".format(i+1))

    rows, cols = imps[0].shape
    x = range(cols)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

    for i in range(num_imps):
        axes[i].plot(x, imps[i][frame_index], x, imps[i][frame_index+frame_shift])
        axes[i].set_title('class: {}, {}'.format(class_name, notes[i]))
        axes[i].set_xlabel('electrod index')
        axes[i].set_ylabel('impedances (Ohm)')
    
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
    axes[0].set_ylabel('impedances (Ohm)')
    col_name1 = "imp_tx_{}_rx_{}_img".format(tx, rx)
    col_name2 = "imp_tx_{}_rx_{}_img".format(rx, tx)
  
    df[[col_name1, col_name2]].plot(ax=axes[1])
    axes[1].set_xlabel('time index')
    axes[1].set_ylabel('impedances (Ohm)')


