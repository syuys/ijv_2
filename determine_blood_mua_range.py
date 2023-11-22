#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:57:56 2020

@author: md703

Objective:
    To investigate the appropriate mua range in ijv & cca
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import Color
import scienceplots
plt.style.use(['science', 'grid'])
# plt.close("all")
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 600

# %% parameters
# read data
observeWl = np.linspace(650, 950, num=301, dtype=int)
projectWl = np.linspace(725, 875, num=151, dtype=int)
epsilonHbO2HbPath = "shared_files/model_input_related/optical_properties/blood/mua/epsilon_hemoglobin.txt"
conc = 150  # [g/L], concentration of hemoglobin (100% for Hb or HbO2)
molecularweightHbO2 = 64532  # [g/mol]
molecularweightHb = 64500  # [g/mol]
differenceThold = 4e2

# analyze SO2
so2Set = np.linspace(0.4, 1, 7)


# %% read epsilon
epsilonHbO2Hb = pd.read_csv(epsilonHbO2HbPath, sep="\t", names=["wl", "HbO2", "Hb"])
paralWl = epsilonHbO2Hb["wl"].values
epsilonHbO2 = epsilonHbO2Hb["HbO2"].values
epsilonHb = epsilonHbO2Hb["Hb"].values


# %% observe wavelength range
# interpolate epsilon to our observe wl
epsilonHbO2Used = np.interp(observeWl, paralWl, epsilonHbO2)  # [cm-1/M]
epsilonHbUsed = np.interp(observeWl, paralWl, epsilonHb)  # [cm-1/M]

# calculate mua from epsilon
muaHbO2 = 2.303 * epsilonHbO2Used * (conc / molecularweightHbO2)  # [1/cm]
muaHb = 2.303 * epsilonHbUsed * (conc / molecularweightHb)  # [1/cm]

# plot epsilon and its difference between HbO2 and Hb
plt.plot(observeWl, epsilonHbO2Used, label="HbO2")
plt.plot(observeWl, epsilonHbUsed, label="Hb")
plt.plot(observeWl, abs(epsilonHbO2Used-epsilonHbUsed), "k--", label="|HbO2-Hb|")
legalWl = observeWl[np.where(abs(epsilonHbO2Used-epsilonHbUsed) > differenceThold)]
legalWlDiff = np.diff(legalWl)
plt.axhline(y=differenceThold, color="gray", linestyle="--", 
            label="threshold={:.1e}\nwl -- [{} {}], [{} {}]".format(differenceThold, legalWl[0], 
                                                                    legalWl[np.where(legalWlDiff>1)[0][0]], 
                                                                    legalWl[np.where(legalWlDiff>1)[0][0]+1], legalWl[-1]))
plt.yscale("log")
plt.legend()
plt.xlabel("wavelength [nm]")
plt.ylabel("epsilon [cm-1/M]")
plt.title("epsilon of HbO2 & Hb")
plt.show()

# plot mua of HbO2 and Hb
plt.plot(observeWl, muaHbO2, label="HbO2")
plt.plot(observeWl, muaHb, label="Hb")
plt.plot(observeWl, abs(muaHbO2-muaHb), "k--", label="|HbO2-Hb|")
plt.axvline(x=725, c="gray")
plt.axvline(x=875, c="gray")
plt.legend()
plt.xlabel("wavelength [nm]")
plt.ylabel("mua [1/cm]")
plt.title("mua of HbO2, Hb")
plt.show()


# %% investigate the appropriate mua range in blood (ijv & cca) from Parah's data
# now we've determined the project wavelength, interpolate again.
epsilonHbO2Used = np.interp(projectWl, paralWl, epsilonHbO2)  # [cm-1/M]
epsilonHbUsed = np.interp(projectWl, paralWl, epsilonHb)  # [cm-1/M]
# calculate mua from epsilon among different sO2 and different hct
concSet = np.linspace(138, 174, 3)
muaHbO2Set = 2.303 * epsilonHbO2Used * (concSet[:, None] / molecularweightHbO2)  # [1/cm]
muaHbSet = 2.303 * epsilonHbUsed * (concSet[:, None] / molecularweightHb)  # [1/cm]
muaWholeBloodSet = muaHbO2Set * so2Set[:, None, None] + muaHbSet * (1-so2Set[:, None, None])  # [1/cm]
# visualization
colorSet = list(Color("lightcoral").range_to(Color("darkred"), so2Set.size))
for i in range(muaWholeBloodSet.shape[0]):
    for j in range(muaWholeBloodSet[i].shape[0]):
        if j == muaWholeBloodSet[i].shape[0]-1:
            plt.plot(projectWl, muaWholeBloodSet[i][j], c=colorSet[i].get_hex(), label=np.round(so2Set[i], 1))
        else:
            plt.plot(projectWl, muaWholeBloodSet[i][j], c=colorSet[i].get_hex())
plt.legend()
plt.xlabel("wl [nm]")
plt.ylabel("mua [1/cm]")
plt.title("mua in different SO2, conc={}".format(concSet))
plt.show()


# %% plot Parah's hemoglobin absorption spec
wledge = [650, 950]
wl = epsilonHbO2Hb["wl"][(epsilonHbO2Hb["wl"]>=wledge[0]) & (epsilonHbO2Hb["wl"]<=wledge[1])]
hbo2 = epsilonHbO2Hb["HbO2"][(epsilonHbO2Hb["wl"]>=wledge[0]) & (epsilonHbO2Hb["wl"]<=wledge[1])]
hb = epsilonHbO2Hb["Hb"][(epsilonHbO2Hb["wl"]>=wledge[0]) & (epsilonHbO2Hb["wl"]<=wledge[1])]
plt.figure(figsize=(4, 2.8))
plt.plot(wl, hbo2, c="red", linewidth=3, label="O$_2$Hb")
plt.plot(wl, hb, c="blue", linewidth=3, label="HHb")
plt.legend(edgecolor="black", fontsize="medium")
plt.grid(visible=False)
# plt.yscale("log")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Molar extinction coefficient [cm$^{-1}$/M]")
plt.show()



