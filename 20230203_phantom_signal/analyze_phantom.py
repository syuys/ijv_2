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
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

# %% parameter
phantomName = "I"
figLog = False
sdsStartIdxSet = [0,  2, 3]
# sdsEndIdxSet =   [-1, -1, -1]
expType = os.path.join("snr_measure", phantomName)
wlStartIdx = 5  #  520
wlEndIdx = -5   # -355
fileSet = glob(os.path.join(expType, "*"))
c1 = "red"
c2 = "blue"
fileSet.sort(key=lambda x: int("".join(map(str, (map(ord, x.split("_")[2]))))))

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
sdsSet = []
for idx, file in enumerate(fileSet[:-1]):
    measureType = "_".join(file.split("/")[-1].split("_")[:2])
    sdsSet.append(int(measureType.split("_")[-1]))
    
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


# analyze
for sdsStartIdx in sdsStartIdxSet:
    targetSdsSet = sdsSet[sdsStartIdx:]
    
    # n for gradient plot
    n = len(targetSdsSet)
    for _ in range(2):
        meanSet = []        
        snrSet = []
        cvSet = []
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        for idx, sds in enumerate(targetSdsSet):
            measureType = f"sds_{sds}"
            data = Signal[measureType]
            if _ == 1:
                data = data - Signal["background"].mean(axis=0)  # signal - background
            
            mean = data.mean(axis=0)
            meanSet.append(mean)
            std = data.std(axis=0, ddof=1)
            snr = mean / std
            cv = std / mean
            snrSet.append(snr)
            cvSet.append(cv)
            
            axes[0].plot(wl, mean, color=colorFader(c1, c2, idx/(n-1)), label=measureType)
            axes[0].set_xlabel("wl [nm]")
            axes[0].set_ylabel("intensity [-]")
            
            axes[1].plot(wl, snr, color=colorFader(c1, c2, idx/(n-1)), label=measureType)
            axes[1].set_xlabel("wl [nm]")
            axes[1].set_ylabel("snr [-]")
        
        if figLog:
            for ax in axes:
                ax.set_yscale("log")
        axes[0].legend(fontsize=8)
        axes[0].set_title("signal")
        axes[1].legend(fontsize=8)
        axes[1].set_title("snr = mean / std")
        if _ == 0:
            fig.suptitle(f"Phantom {phantomName}, sds {targetSdsSet[0]}mm to {targetSdsSet[-1]}mm")
        else:
            fig.suptitle(f"Phantom {phantomName}, sds {targetSdsSet[0]}mm to {targetSdsSet[-1]}mm (SUBTRACT BACKGROUND)")
        plt.tight_layout()
        plt.show()
        
        if _ == 1:
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            for idx, sds in enumerate(targetSdsSet):
                axes[0].scatter(meanSet[idx], snrSet[idx], s=8, color=colorFader(c1, c2, idx/(n-1)), label=f"sds_{sds}")
                axes[1].scatter(meanSet[idx], cvSet[idx], s=8, color=colorFader(c1, c2, idx/(n-1)), label=f"sds_{sds}")
            axes[0].set_xlabel("intensity [-]")
            axes[0].set_ylabel("snr [-]")
            axes[1].set_xlabel("intensity [-]")
            axes[1].set_ylabel("cv [-]")
            vals = axes[1].get_yticks()
            axes[1].set_yticklabels(['{:,.1%}'.format(x) for x in vals])
            axes[0].legend()
            axes[1].legend()
            axes[0].set_title("intensity v.s. snr")
            axes[1].set_title("intensity v.s. cv")
            fig.suptitle(f"sds {sdsSet[0]}mm to {sdsSet[-1]}mm (substract background)")
            plt.tight_layout()
            plt.show()

