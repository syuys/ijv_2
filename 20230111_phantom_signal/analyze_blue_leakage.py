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
figLog = False
expType = "blue_light_leakage"
wlStartIdx = 520  #  520
wlEndIdx = -355   # -355
fileSet = glob(os.path.join(expType, "*"))
c1 = "red"
c2 = "blue"
fileSet.sort(key=lambda x: int("".join(map(str, (map(ord, x.split("_")[6]))))))
def colorFader(c1,c2,mix=0): # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


# %% analyze signal

# access wavelength and background
df = pd.read_csv(fileSet[-1], sep="\t", skiprows=14, 
                  usecols = lambda x: "Unnamed" not in x)
print(f"{fileSet[-1]}: \n{df.shape}\n")
wl = df.columns.values.astype(float)[wlStartIdx:wlEndIdx]
bg = df.to_numpy()[:, wlStartIdx:wlEndIdx]
bgSNR = bg.mean(axis=0) / bg.std(axis=0, ddof=1)
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(wl, bg.mean(axis=0))
axes[0].set_xlabel("wl [nm]")
axes[0].set_ylabel("intensity [-]")
axes[1].plot(wl, bgSNR)
axes[1].set_xlabel("wl [nm]")
axes[1].set_ylabel("snr [-]")
axes[0].set_title("backround mean")
axes[1].set_title("background snr = mean / std")
plt.tight_layout()
plt.show()

# access other signal
n = len(fileSet)
sdsSet = []
meanSet = []
stdSet = []
snrSet = []
for _ in range(2):
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for idx, file in enumerate(fileSet[:-1]):
        measureType = "_".join(file.split("/")[-1].split("_")[:5])
        
        df = pd.read_csv(file, sep="\t", skiprows=14, 
                          usecols = lambda x: "Unnamed" not in x)
        print(f"{file}: \n{df.shape}\n")
        
        df = df.to_numpy()[:, wlStartIdx:wlEndIdx]
        if _ == 1:
            df = df - bg  # signal - background
        #     zorder=idx
        # else:
        #     zorder=len(fileSet) - idx
        mean = df.mean(axis=0)
        meanSet.append(mean)
        std = df.std(axis=0, ddof=1)
        stdSet.append(std)
        snr = mean / std
        snrSet.append(snr)
        
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

# snrSet = np.array(snrSet)
# for idx in range(102, 1000, 103):
#     plt.plot(sdsSet, snrSet[:, idx], linestyle="-", marker=".", label=np.around(wl[idx]))
#     # plt.plot(snrSet[:, idx], label=wl[idx])
#     plt.xlabel("sds [mm]")
#     plt.ylabel("snr [-]")
# plt.legend()
# plt.title("Variation w.r.t Different Wl")
# plt.show()