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
tissueType = ["skin", "fat", "muscle", "blood"]
wlProjectStart = 725
wlProjectEnd = 875
wlProject = np.linspace(wlProjectStart, wlProjectEnd, num=151)
bloodConcSet = np.array([138, 174])
bloodSO2Set = {"ijv": np.linspace(0.4, 0.8, 5),
               "cca": np.linspace(0.9, 1.0, 2)
               }  # for analysis of blood SO2


# %% main
# read raw data, do interpolation, and plot
rawData = {}  # raw data
interpData = {}  # processed data
for tissue in tissueType:
    if tissue == "blood":
        epsilonHbO2HbPath = "blood/mua/epsilon_hemoglobin.txt"
        epsilonHbO2Hb = pd.read_csv(epsilonHbO2HbPath, sep="\t", names=["wl", "HbO2", "Hb"])
        cs = CubicSpline(epsilonHbO2Hb.wl.values, epsilonHbO2Hb.HbO2.values, extrapolate=False)
        epsilonHbO2Used = cs(wlProject)  # [cm-1/M]
        cs = CubicSpline(epsilonHbO2Hb.wl.values, epsilonHbO2Hb.Hb.values, extrapolate=False)
        epsilonHbUsed = cs(wlProject)  # [cm-1/M]
        muaHbO2Set = 2.303 * epsilonHbO2Used * (bloodConcSet[:, None] / 64532)  # [1/cm]
        muaHbSet = 2.303 * epsilonHbUsed * (bloodConcSet[:, None] / 64500)  # [1/cm]
        for key in bloodSO2Set.keys():
            interpData[key] = {}
            muaWholeBloodSet = muaHbO2Set * bloodSO2Set[key][:, None, None] + muaHbSet * (1-bloodSO2Set[key][:, None, None])  # [1/cm]
            # visualization
            colorSet = list(Color("lightcoral").range_to(Color("darkred"), bloodSO2Set[key].size))
            for i in range(muaWholeBloodSet.shape[0]):
                for j in range(muaWholeBloodSet[i].shape[0]):
                    if j == muaWholeBloodSet[i].shape[0]-1:
                        plt.plot(wlProject, muaWholeBloodSet[i][j], c=colorSet[i].get_hex(), label=np.round(bloodSO2Set[key][i], 1))
                    else:
                        plt.plot(wlProject, muaWholeBloodSet[i][j], c=colorSet[i].get_hex())
            plt.legend()
            plt.xlabel("wl [nm]")
            plt.ylabel("mua [1/cm]")
            plt.title("{}'s mua in different SO2, conc={}".format(key, bloodConcSet))
            plt.show()
            interpData[key]["Parah's data"] = muaWholeBloodSet
    else:
        interpData[tissue] = {}
        muaPathSet = glob(os.path.join(tissue, "mua", "*.csv"))
        for muaPath in muaPathSet:
            name = muaPath.split("/")[-1].replace(".csv", "")        
            # read raw data
            df = pd.read_csv(muaPath)
            # rawData[tissue][name] = df        
            # plot raw data
            plt.plot(df.wavelength.values, df.mua.values, label=name)
            # interpolate to wl-project and save
            cs = CubicSpline(df.wavelength.values, df.mua.values, extrapolate=False)
            interpData[tissue][name] = cs(wlProject)
        plt.legend(fontsize="x-small")
        plt.xlabel("wavelength [nm]")
        plt.ylabel("mua [1/cm]")
        plt.title(tissue + "'s mua raw data")
        plt.show()

# plot interpolated data
for tissue in tissueType:
    if tissue != "blood":
        for source, data in interpData[tissue].items():
            plt.plot(wlProject, data, label=source)
        plt.legend(fontsize="x-small")
        plt.xlabel("wavelength [nm]")
        plt.ylabel("mua [1/cm]")
        plt.title(tissue + "'s mua interp data")
        plt.show()
        
# show mua upper bound and lower bound and save to .json file
muaRange = {}
muaRange["__comment__"] = "The mua upper bound and lower bound below are all in unit of [1/cm]."
for tissue in interpData.keys():
    allmua = np.array(list(interpData[tissue].values()))
    muaRange[tissue] = [np.floor(np.nanmin(allmua)*1e3)/1e3, np.ceil(np.nanmax(allmua)*1e3)/1e3]
    print(f"{tissue}'s mua ---> min={np.nanmin(allmua)}, max={np.nanmax(allmua)}")
    
with open("mua_bound.json", "w") as f:
    json.dump(muaRange, f, indent=4)