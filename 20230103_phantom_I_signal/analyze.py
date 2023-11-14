#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:38:30 2023

@author: md703
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

fileNameSet = ["test_QEP039351__0__21-22-02-780.txt"
               ]
# bg = pd.read_csv("SDSBG_QEP039351__0__23-23-39-339.txt", 
#                  sep='\t', header=None, skiprows=15, 
#                  usecols = [i for i in range(2, 1046)])
# # bg = bg.to_numpy()[:, 5:-5]
# bg = bg.to_numpy()

sdsSet = []
meanSet = []
stdSet = []
snrSet = []
fig, ax = plt.subplots(1, 2, figsize=(11,3))
for fileName in fileNameSet:
    measureType = "1010"
    sdsSet.append(float(measureType))
    wl = pd.read_csv(fileName, sep=" ", skiprows=13, header=None)
    wl = wl[0][1].split("\t")
    while("" in wl):
        wl.remove("")
    # wl = np.array(wl).astype(float)[5:-5]
    wl = np.array(wl).astype(float)
    
    df = pd.read_csv(fileName, sep='\t', header=None, skiprows=15, 
                     usecols = [i for i in range(2, 1046)])
    # df = df.to_numpy()[:188, 5:-5]
    df = df.to_numpy()[:188]
    # df = df - bg  # signal - background
    mean = df.mean(axis=0)
    meanSet.append(mean)
    std = df.std(axis=0, ddof=1)
    stdSet.append(std)
    snr = mean / std
    snrSet.append(snr)
    
    ax[0].plot(wl, mean, label=measureType)
    ax[0].set_xlabel("wl [nm]")
    ax[0].set_ylabel("reflectance [-]")
    
    ax[1].plot(wl, snr, label=measureType)
    ax[1].set_xlabel("wl [nm]")
    ax[1].set_ylabel("snr [-]")

ax[0].legend(fontsize=7)
ax[0].set_title("Signal - Background")
ax[1].legend(fontsize=7)
ax[1].set_title("SNR = Mean / Std")
plt.show()

snrSet = np.array(snrSet)
for idx in range(102, 1000, 103):
    plt.plot(sdsSet, snrSet[:, idx], label=wl[idx])
    plt.xlabel("sds [mm]")
    plt.ylabel("snr [-]")
plt.legend()
plt.title("Variation w.r.t Different Wl")
plt.show()