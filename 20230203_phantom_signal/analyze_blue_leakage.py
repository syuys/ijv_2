#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:38:30 2023

@author: md703
"""

import numpy as np
import pandas as pd
import os
from glob import glob
import scipy.signal as signal
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

# %% parameter
figLog = False
expType = "clay"
wlStartIdx = 5  #  520
wlEndIdx = -5   # -355
fileSet = glob(os.path.join(expType, "*"))
c1 = "red"
c2 = "blue"
fileSet.sort(key=lambda x: int("".join(map(str, (map(ord, x.split("_")[4]))))))

def colorFader(c1,c2,mix=0): # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def plot_info(data, name):
    snr = data.mean(axis=0) / data.std(axis=0, ddof=1)
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].plot(wl, data.mean(axis=0))
    axes[0].set_xlabel("wl [nm]")
    axes[0].set_ylabel("intensity [-]")
    axes[1].plot(wl, snr)
    axes[1].set_xlabel("wl [nm]")
    axes[1].set_ylabel("snr [-]")
    axes[0].set_title(f"{name}, mean")
    axes[1].set_title(f"{name}, snr = mean / std")
    plt.tight_layout()
    plt.show()

def medianfilt(data, k):
    if data.ndim == 1:
        data = signal.medfilt(data, kernel_size=k)
    if data.ndim == 2:
        for idx in range(data.shape[0]):
            data[idx] = signal.medfilt(data[idx], kernel_size=k)
    return data


# %% analyze signal
Signal = {}

# access wavelength and background
df = pd.read_csv(fileSet[-1], sep="\t", skiprows=14, 
                  usecols = lambda x: "Unnamed" not in x)
print(f"{fileSet[-1]}: \n{df.shape}\n")
wl = df.columns.values.astype(float)[wlStartIdx:wlEndIdx]
Signal["background"] = df.to_numpy()[:, wlStartIdx:wlEndIdx]

plot_info(Signal["background"], "Background")

# access other signal
for idx, file in enumerate(fileSet[:-1]):
    measureType = "_".join(file.split("/")[-1].split("_")[:5])
    
    df = pd.read_csv(file, sep="\t", skiprows=14, 
                      usecols = lambda x: "Unnamed" not in x)
    print(f"{file}: \n{df.shape}\n")
    
    Signal[measureType] = df.to_numpy()[:, wlStartIdx:wlEndIdx]

# check impulse (salt and pepper noise)
for measureType, data in Signal.items():
    for d in data:
        plt.plot(wl, d)
    plt.xlabel("wl [nm]")
    plt.ylabel("intensity [-]")
    plt.title(f"{measureType}")
    plt.show()        

# delete data (or add median filter to data) which have impulse and do final check
Signal["background"] = np.delete(Signal["background"], np.argwhere(Signal["background"].max(axis=1)>150), axis=0)
# plt.plot(wl, Signal["background"][37])
# plt.show()
# plt.plot(wl, medianfilt(Signal["background"][37], 11))
# plt.show()
for data in Signal["background"]:
    plt.plot(wl, data)
plt.xlabel("wl [nm]")
plt.ylabel("intensity [-]")
plt.title("background after removing bad signal")
plt.show()

plot_info(Signal["background"], "Background")


# analyze
n = len(fileSet) -1
for _ in range(2):
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for idx, (measureType, data) in enumerate(list(Signal.items())[1:]):
        if _ == 1:
            data = data - Signal["background"].mean(axis=0)  # signal - background
        
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)
        snr = mean / std
        
        axes[0].plot(wl, mean, color=colorFader(c1, c2, idx/(n-1)), label=measureType)
        axes[0].set_xlabel("wl [nm]")
        axes[0].set_ylabel("intensity [-]")
        
        axes[1].plot(wl, snr, color=colorFader(c1, c2, idx/(n-1)), label=measureType)
        axes[1].set_xlabel("wl [nm]")
        axes[1].set_ylabel("snr [-]")
    
    if figLog:
        for ax in axes:
            ax.set_yscale('log')
    axes[0].legend()
    axes[0].set_title("signal")
    axes[1].legend()
    axes[1].set_title("snr = mean / std")
    if _ == 0:
        fig.suptitle("Raw Signal")
    else:
        fig.suptitle("Substract Background")
    plt.tight_layout()
    plt.show()


# normalize signal and compare
for idx, (measureType, data) in enumerate(list(Signal.items())[1:]):
    data = data - Signal["background"].mean(axis=0)  # signal - background
    
    mean = data.mean(axis=0)
    normalizedMean = mean / mean.mean()
    plt.plot(wl, normalizedMean, color=colorFader(c1, c2, idx/(n-1)), label=measureType)
plt.xlabel("wl [nm]")
plt.ylabel("normalized intensity [-]")
plt.legend()
plt.title("Comparison")
plt.show()







