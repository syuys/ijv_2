#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:45:08 2023

@author: md703
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import os
from glob import glob
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

# %% parameters
muaPath = "coefficients_OxyDeoxyConc20.csv"
epsilonHbO2HbPath = "blood/mua/epsilon_hemoglobin.txt"
conc = 150  # [g/L]
molecularweightHbO2 = 64532  # [g/mol]
molecularweightHb = 64500  # [g/mol]
mua = pd.read_csv(muaPath, usecols = lambda x: "Unnamed" not in x)
targetWl = mua["wavelength"].values  # mua["wavelength"].values, np.linspace(725, 850, 126)
issave = False

# %% main

# fat mua
fatMuaPathSet = glob(os.path.join("fat", "mua", "*.csv"))
fatMua = []
for fatMuaPath in fatMuaPathSet:
    name = os.path.split(fatMuaPath)[-1]       
    # read raw data
    df = pd.read_csv(fatMuaPath)  
    # plot raw data
    plt.plot(df.wavelength.values, df.mua.values, label=name, marker="o")
    # interpolate to wl-project and save
    cs_fat = CubicSpline(df.wavelength.values, df.mua.values, extrapolate=False)
    interpMua = cs_fat(targetWl)
    fatMua.append(interpMua)
    plt.plot(targetWl, interpMua, label=f"{name} - fit", linestyle="--")
fatMua = np.array(fatMua)
fatMuaMean = fatMua.mean(axis=0)
plt.plot(targetWl, fatMuaMean, label="fit - average", linestyle="--", color="gray")
plt.legend()
plt.xlabel("wl [nm]")
plt.ylabel("mua [1/cm]")
plt.title("fat mua")
plt.show()

# blood mua
# read epsilon
epsilonHbO2Hb = pd.read_csv(epsilonHbO2HbPath, sep="\t", names=["wl", "HbO2", "Hb"])
wl = epsilonHbO2Hb["wl"].values
epsilonHbO2 = epsilonHbO2Hb["HbO2"].values
epsilonHb = epsilonHbO2Hb["Hb"].values

# interpolate epsilon to our target wl
epsilonHbO2Used = np.interp(targetWl, wl, epsilonHbO2)  # [cm-1/M]
epsilonHbUsed = np.interp(targetWl, wl, epsilonHb)  # [cm-1/M]

# calculate mua from epsilon
HbO2_mua = 2.303 * epsilonHbO2Used * (conc / molecularweightHbO2)  # [1/cm] - from Prahl
Hb_mua = 2.303 * epsilonHbUsed * (conc / molecularweightHb)  # [1/cm] - from Prahl

#%% plot
plt.plot(targetWl, HbO2_mua, label="HbO2")
plt.plot(targetWl, Hb_mua, label="Hb")
plt.xlabel("wl [nm]")
plt.ylabel("mua [1/cm]")
plt.legend()
plt.title("Prahl")
plt.show()

plt.plot(targetWl, np.interp(targetWl, mua["wavelength"].values, mua["oxy"].values), label="HbO2")
plt.plot(targetWl, np.interp(targetWl, mua["wavelength"].values, mua["deoxy"].values), label="Hb")
plt.xlabel("wl [nm]")
plt.ylabel("mua [1/cm]")
plt.legend()
plt.title("Toast coefficients.csv")
plt.show()

plt.plot(targetWl, np.interp(targetWl, mua["wavelength"].values, mua["fat"].values))
plt.xlabel("wl [nm]")
plt.ylabel("mua [1/cm]")
plt.title("Toast coefficients.csv -- fat")
plt.show()

# %% save  -- all species' mua unit: 1/cm
mua["oxy"] = HbO2_mua
mua["deoxy"] = Hb_mua
mua["fat"] = fatMuaMean
mua = mua.drop(columns="musp")
if issave:
    mua.to_csv(f"coefficients_in_cm-1_OxyDeoxyConc{conc}.csv", index=False)

# final check
for species in mua.columns.values[1:]:
    plt.plot(targetWl, mua[species].values)
    plt.xlabel("wl [nm]")
    plt.ylabel("mua [1/cm]")
    plt.title(f"Final coefficients - {species}")
    plt.show()