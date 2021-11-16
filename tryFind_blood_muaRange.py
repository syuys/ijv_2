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
plt.close("all")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

# %% read epsilon
targetWl = np.linspace(650, 950, num=401, dtype=int)
epsilonHbO2HbPath = "shared_files/model_input_related/absorption/epsilon_hemoglobin.txt"
toastMuaPath = "20210828_mcxnewcode_reliability_validation/input/coefficients.csv"
# concentration of hemoglobin (100% for Hb or HbO2)
conc = 110  # [g/L]
molecularweightHbO2 = 64532  # [g/mol]
molecularweightHb = 64500  # [g/mol]

# %% read epsilon & calculate mua & plot
# read toast's mua
toastMua = pd.read_csv(toastMuaPath)

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

# plot epsilon
plt.plot(targetWl, epsilonHbO2Used, label="HbO2")
plt.plot(targetWl, epsilonHbUsed, label="Hb")
plt.plot(targetWl, abs(epsilonHbO2Used-epsilonHbUsed), "k--", label="|HbO2-Hb|")
differenceThold = 4e2
legalWl = targetWl[np.where(abs(epsilonHbO2Used-epsilonHbUsed) > differenceThold)]
legalWlDiff = np.diff(legalWl)
plt.axhline(y=differenceThold, color="gray", linestyle="--", label="threshold={:.1e}\nwl -- [{} {}], [{} {}]".format(differenceThold, 
                                                                                                                     legalWl[0],
                                                                                                                     legalWl[np.where(legalWlDiff>1)[0][0]],
                                                                                                                     legalWl[np.where(legalWlDiff>1)[0][0]+1],
                                                                                                                     legalWl[-1]))
plt.yscale("log")
plt.legend()
plt.xlabel("wavelength [nm]")
plt.ylabel("epsilon [cm-1/M]")
plt.title("epsilon of HbO2 & Hb")
plt.show()

# plot mua
plt.plot(targetWl, HbO2_mua, label="HbO2")
plt.plot(targetWl, Hb_mua, label="Hb")
plt.plot(targetWl, abs(HbO2_mua-Hb_mua), "k--", label="|HbO2-Hb|")
# plt.plot(toastMua.wavelength.values, toastMua.water.values, label="water (Tu Shi Chen's data)")
plt.axvline(x=725, c="gray")
plt.axvline(x=875, c="gray")
plt.legend()
plt.xlabel("wavelength [nm]")
plt.ylabel("mua [1/cm]")
plt.title("mua of HbO2, Hb, Water")
plt.show()

# plot toast's mua
plt.plot(toastMua.wavelength.values, toastMua.oxy.values, label="HbO2")
plt.plot(toastMua.wavelength.values, toastMua.deoxy.values, label="Hb")
plt.plot(toastMua.wavelength.values, abs(toastMua.oxy.values-toastMua.deoxy.values), "k--", label="|HbO2-Hb|")
plt.plot(toastMua.wavelength.values, toastMua.water.values, label="water")
plt.legend()
plt.xlabel("wavelength [nm]")
plt.ylabel("mua [1/cm]")
plt.title("mua of HbO2, Hb, Water (Tu Shi Chen's data)")
plt.show()

# # %% toast data
# mua_toast = pd.read_csv(
#     "/home/md703/Desktop/ijv/input/coefficients.csv").drop(columns="Unnamed: 0", axis=1)

# # 1/cm
# HbO2_mua_toast = np.interp(
#     targetWl, mua_toast["wavelength"].values, mua_toast["oxy"].values)

# # 1/cm
# Hb_mua_toast = np.interp(
#     targetWl, mua_toast["wavelength"].values, mua_toast["deoxy"].values)

# plt.plot(targetWl, HbO2_mua_toast, label="HbO2_toast")
# plt.plot(targetWl, Hb_mua_toast, label="Hb_toast")
# plt.legend()
# plt.xlabel("wl [nm]")
# plt.ylabel("mua [1/cm]")
# plt.title("mua of HbO2 & Hb (toast)")
# plt.show()

# # %% calculate and plot tissue's mua (from toast data)
# # ijv SO2
# SjO2 = 0.7
# # 1/cm
# mua_ijv = HbO2_mua_toast * SjO2 + Hb_mua_toast * (1-SjO2)
# # 1/cm --> 1/mm
# mua_ijv = mua_ijv * 0.1

# # cca SO2
# SaO2 = 0.98
# # 1/cm
# mua_cca = HbO2_mua_toast * SaO2 + Hb_mua_toast * (1-SaO2)
# # 1/cm --> 1/mm
# mua_cca = mua_cca * 0.1

# # plot
# plt.figure(dpi=200)
# plt.plot(targetWl, mua_ijv, '-o',
#          label="{:.0%} ijv, range=({:.2f}, {:.2f})".format(SjO2, mua_ijv.min(), mua_ijv.max()))
# plt.plot(targetWl, mua_cca, '-o',
#          label="{:.0%} cca, range=({:.2f}, {:.2f})".format(SaO2, mua_cca.min(), mua_cca.max()))
# plt.legend()
# plt.xlabel("wl [nm]")
# plt.ylabel("mua [1/mm]")
# plt.title("mua of ijv & cca (toast)")
# plt.show()

# %% calculate and plot tissue's mua (from Prahl data)
# ijv SO2
SjO2 = 0.7
# 1/cm
mua_ijv = HbO2_mua * SjO2 + Hb_mua * (1-SjO2)
# 1/cm --> 1/mm
mua_ijv = mua_ijv * 0.1

# cca SO2
SaO2 = 1
# 1/cm
mua_cca = HbO2_mua * SaO2 + Hb_mua * (1-SaO2)
# 1/cm --> 1/mm
mua_cca = mua_cca * 0.1

# plot
plt.figure(dpi=200)
plt.plot(targetWl, mua_ijv, '-o',
         label="{:.0%} ijv, range=({:.2f}, {:.2f})".format(SjO2, mua_ijv.min(), mua_ijv.max()))
plt.plot(targetWl, mua_cca, '-o',
         label="{:.0%} cca, range=({:.3f}, {:.2f})".format(SaO2, mua_cca.min(), mua_cca.max()))
plt.legend()
plt.xlabel("wl [nm]")
plt.ylabel("mua [1/mm]")
plt.title("mua of ijv & cca (Prahl, conc={}g/L)".format(conc))
plt.show()

# %% find mua range of each SO2 -- blood
mua = {"HbO2": {"Prahl": HbO2_mua,
                "toast": HbO2_mua_toast
                },
       "Hb": {"Prahl": Hb_mua,
              "toast": Hb_mua_toast}
       }

# SO2
SO2 = np.arange(0.5, 1.01, step=0.025)
# 1/cm
source = "Prahl"
mua_blood = mua["HbO2"][source][:, None] * SO2[None, :] + \
    mua["Hb"][source][:, None] * (1-SO2)[None, :]
# 1/cm --> 1/mm
mua_blood = mua_blood * 0.1

# plot
plt.figure(figsize=(15, 5), dpi=200)
for idx, value in enumerate(SO2):
    ymin = mua_blood.min(axis=0)[idx]
    ymax = mua_blood.max(axis=0)[idx]
    plt.axvline(x=value, ymin=ymin, ymax=ymax, lw=5)

plt.grid()
plt.xticks(SO2)
plt.xlabel("blood SO2")
plt.ylabel("mua range [1/mm]")
if source == "Prahl":
    plt.title("blood mua variation (source={}, conc={}g/L)".format(source, conc))
else:
    plt.title("blood mua variation (source={})".format(source))
plt.show()
