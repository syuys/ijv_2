#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:33:37 2021

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from colour import Color
import matplotlib.pyplot as plt
plt.close("all")
import json
import os
from glob import glob
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% parameters
tissueType = ["blood"]
wlProjectStart = 725
wlProjectEnd = 875
wlProject = np.linspace(wlProjectStart, wlProjectEnd, num=151)
so2Set = np.linspace(0.4, 1, 7)  # for analysis of blood SO2


# %% main
# read raw data, do interpolation, and plot
rawData = {}  # raw data
interpData = {}  # processed data
for tissue in tissueType:
    rawData[tissue] = {}
    interpData[tissue] = {}
    muaPathSet = glob(os.path.join(tissue, "mua", "*.csv"))
    for muaPath in muaPathSet:
        name = muaPath.split("/")[-1].replace(".csv", "")        
        # read raw data
        df = pd.read_csv(muaPath)
        if tissue == "blood":
            df.mua = df.mua*10  # convert 1/mm to 1/cm when tissue's type is "blood".
        rawData[tissue][name] = df        
        # plot raw data
        plt.plot(df.wavelength.values, df.mua.values, label=name)
        # interpolate to wl-project and save
        cs = CubicSpline(df.wavelength.values, df.mua.values)
        interpData[tissue][name] = cs(wlProject)
    plt.yscale("log")
    # plt.legend()
    plt.xlabel("wavelength [nm]")
    plt.ylabel("mua [1/cm]")
    plt.title(tissue + "'s mua raw data")
    plt.show()

for tissue in tissueType:
    for source, data in interpData[tissue].items():
        plt.plot(wlProject, data, label=source)
    plt.legend(fontsize="x-small")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("mua [1/cm]")
    plt.title(tissue + "'s mua interp data")
    plt.show()
        
muaHbO2Set = np.vstack((interpData["blood"]['Friebel et al, oxgenated blood (Hct=41.3 _)'],
                        interpData["blood"]['Friebel et al, oxgenated blood (Hct=52.1 _)'],
                        interpData["blood"]['Roggan et al, oxgenated blood(Hct=41.3 _)'],
                        interpData["blood"]['Roggan et al, oxgenated blood(Hct=52.1 _)']))
muaHbSet = np.vstack((interpData["blood"]['Friebel et al, de-oxgenated blood (Hct=41.3 _)'],
                      interpData["blood"]['Friebel et al, de-oxgenated blood (Hct=52.1 _)'],
                      interpData["blood"]['Roggan et al, de-oxgenated blood(Hct=41.3 _)'],
                      interpData["blood"]['Roggan et al, de-oxgenated blood(Hct=52.1 _)']))
muaWholeBloodSet = muaHbO2Set * so2Set[:, None, None] + muaHbSet * (1-so2Set[:, None, None])  # [1/cm]
# visualization
colorSet = list(Color("lightcoral").range_to(Color("darkred"), so2Set.size))
for i in range(muaWholeBloodSet.shape[0]):
    for j in range(muaWholeBloodSet[i].shape[0]):
        if j == muaWholeBloodSet[i].shape[0]-1:
            plt.plot(wlProject, muaWholeBloodSet[i][j], c=colorSet[i].get_hex(), label=np.round(so2Set[i], 1))
        else:
            plt.plot(wlProject, muaWholeBloodSet[i][j], c=colorSet[i].get_hex())
plt.legend()
plt.xlabel("wl [nm]")
plt.ylabel("mua [1/cm]")
plt.title("mua in different SO2")
plt.show()
        
        
        
        
        
        