#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:58:59 2023

@author: md703
"""

import numpy as np
import pandas as pd
import os
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% parameters
folderSet = [
        "20230601_HW_contrast_trend_and_holder_stability",
        "20230609_KB_contrast_trend",
        # "20230616_BY_contrast_trend",
        # "20230619_EU_contrast_trend",
        # "20230621_HY_contrast_trend",
    ]
subjectSet = []
simdata = pd.read_csv("sim_contrast_trend.csv")
optSdsIdxSet = [1, 2, 0, 6, 0]

def get_optsds(x, a, b, c, d):
    optsds = a*x**4 + b*x**2 + c*x + d
    return optsds


# %% main
for idx, folder in enumerate(folderSet):
    subject = folder.split("_")[1]
    subjectSet.append(subject)
    sds_con_df = pd.read_csv(os.path.join(folder, f"{subject}_contrast_trend.csv"))
    xlabel = sds_con_df.columns[0]
    ylabel = sds_con_df.columns[1]
    obsRange = sds_con_df[xlabel].values <= 30
    plt.plot(sds_con_df[xlabel].values[obsRange], sds_con_df[ylabel].values[obsRange], 
             marker=".", linestyle="-")
    plt.plot(sds_con_df[xlabel][optSdsIdxSet[idx]], sds_con_df[ylabel][optSdsIdxSet[idx]], 
             "o", color="red")
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(subject)
    plt.show()

xidx = 0
xlabel = simdata.columns[xidx]
x = simdata[xlabel].values
y = simdata["all mus mua 50% - OptSDS [mm]"].values
xfit = np.linspace(x[0], x[-1], 100)
popt, pcov = curve_fit(get_optsds, x, y, maxfev=5000)
yfit = get_optsds(xfit, *popt)
simdata['all mus mua 100% - OptSDS [mm]'] = simdata['all mus mua 100% - OptSDS [mm]'].fillna(0)

# plot invivo
for idx, subject in enumerate(subjectSet):
    with open(os.path.join("ultrasound_image_processing", "202305~", subject, f"{subject}_geo.json")) as f:
        geo = json.load(f)
    ijvupperedge2surf = geo["MiddleNeck"]["IJV"]["Depth"] - geo["MiddleNeck"]["IJV"]["MinorAxisNormal"]
    sds_con_df = pd.read_csv(os.path.join(folderSet[idx], f"{subject}_contrast_trend.csv"))    
    optsds = sds_con_df["SDS [mm]"][optSdsIdxSet[idx]]
    plt.plot(ijvupperedge2surf, optsds, "o", label=subject)
        

# plot sim
plt.plot(x, y, "o", label="Sim - all mus mua 50%")
plt.plot(xfit, yfit, "-", label="fit")
# plt.plot(simdata[xlabel][2], simdata['all mus mua 100% - OptSDS [mm]'][2], "o", 
#           label="Sim - all mus mua 100%")
plt.grid()
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0. )
plt.xlabel(xlabel)
plt.ylabel("Optimal SDS [mm]")
plt.title("Comparison")
plt.show()
    






