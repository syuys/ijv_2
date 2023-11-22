#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:58:59 2023

@author: md703
"""

import numpy as np
import pandas as pd
from glob import glob
import os
import json
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% parameters
simFolderSet = [
        "20230628_contrast_investigate_op_indifferentdepth_sdsrange_5to45_g99",
        "20230505_contrast_investigate_op_sdsrange_5to45_g99",
        "20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99"
    ]
geoBoundDir = "shared_files/model_input_related/geo_bound.json"
judgeContrastEndIdx = 22
stdNumSet = ["-1.6", "-1", "+0", "+1", "+2"]

def get_optsds(x, a, b, c, d):
    optsds = a*x**3 + b*x**2 + c*x + d
    return optsds


# %% organize sim
with open(geoBoundDir) as f:
    geoBound = json.load(f)
ijvdepthAve = geoBound["IJV"]["Depth"]["Average"]
ijvdepthStd = geoBound["IJV"]["Depth"]["Std"]
ijvMinorAxisNormalAve = geoBound["IJV"]["MinorAxisNormal"]["Average"]
ijvupperedge2surfSet = {}
for stdNum in stdNumSet:
    ijvupperedge2surfSet[stdNum] = ijvdepthAve + float(stdNum)*ijvdepthStd - ijvMinorAxisNormalAve

contrastDirSet = []
for idx, folder in enumerate(simFolderSet):
    contrastDirSet += glob(os.path.join(folder, "contrast*.csv"))
contrastDirSet.sort(key=lambda x: (float(x.split("/")[-1].split("_")[2]),
                                   float(x.split("/")[-1].split("_")[5][:-1]),
                                   float(x.split("/")[-1].split("_")[7][:-5]),)
                    )

# plot all kinds of contrast and select contrasts having peak
contrast_df_set = {}
sessionIDSet = []
markSet = {}
for contrastDir in contrastDirSet:
    sessionID = contrastDir.split("/")[-1][9:-4]
    sessionIDSet += sessionID
    contrast_df = pd.read_csv(contrastDir)
    contrast_df_set[sessionID] = contrast_df
    sds = contrast_df["SDS [mm]"]
    contrast = contrast_df["Rmax/Rmin [-]"] - 1
    maxidx = np.argmax(contrast[:judgeContrastEndIdx])
    if judgeContrastEndIdx - maxidx >= 4:
        marksds = list(sds[maxidx-1:maxidx+2])
        markcon = contrast[maxidx-1:maxidx+2]
        markSet[sessionID] = marksds
        plt.plot(marksds, markcon, marker=".", color="red", zorder=1)
    plt.plot(sds, contrast, marker=".", zorder=0)
    plt.grid()
    plt.xlabel("SDS [mm]")
    plt.ylabel("Rmax/Rmin - 1  [-]")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.title(sessionID)
    plt.show()

# plot ijvupperedge2surf v.s. peak sds of selected-contrast
muscolorstyle = {"0%": "tab:blue", 
                 "50%": "tab:orange", 
                 "100%": "tab:red",  
                }
musshiftstyle = {"0%": -0.2, 
                 "50%": 0, 
                 "100%": 0.2,  
                }
muamarkerstyle = {"0%": "x", 
                  "50%": "*", 
                  "100%": ".",              
                  }
lnSet = []
anchor_x = [
        ijvupperedge2surfSet["-1.6"],
        ijvupperedge2surfSet["-1"],
        ijvupperedge2surfSet["+0"],
        ijvupperedge2surfSet["+1"]
    ]
anchor_y = [
        markSet["depth_-1.6_std_mus_50%_mua_50%"][1],
        markSet["depth_-1_std_mus_50%_mua_50%"][1],
        markSet["depth_+0_std_mus_50%_mua_50%"][1],
        markSet["depth_+1_std_mus_50%_mua_50%"][1]
    ]
popt, pcov = curve_fit(get_optsds, anchor_x, anchor_y)
anchor_x_fit = np.linspace(anchor_x[0], anchor_x[-1], 100)
anchor_y_fit = get_optsds(anchor_x_fit, *popt)
for sessionID, marksds in markSet.items():
    stdType = sessionID.split("_")[1]
    ijvupperedge2surf = ijvupperedge2surfSet[stdType]
    mus = sessionID.split('_')[-3]
    mua = sessionID.split('_')[-1]
    if stdType == "-1":
        lnSet += plt.plot(np.repeat(ijvupperedge2surf, len(marksds)) + musshiftstyle[mus],
                         marksds, muamarkerstyle[mua], color=muscolorstyle[mus],
                         label=f"({mus}, {mua})")
    else:
        plt.plot(np.repeat(ijvupperedge2surf, len(marksds)) + musshiftstyle[mus],
                 marksds, muamarkerstyle[mua], color=muscolorstyle[mus],
                 label=f"({mus}, {mua})")
plt.plot(anchor_x_fit, anchor_y_fit, "-", color="tab:orange",
         linewidth=20, alpha=0.3, zorder=0)
plt.grid()
plt.legend(lnSet, [ln.get_label() for ln in lnSet], bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0. )
plt.xlabel("Distance of IJV upper edge to skin surface [mm]")
plt.ylabel("Short-optimal SDS [mm]")
plt.title("Comparison of (mus, mua)")
plt.show()

# %% organize invivo
sdsFitRange = [10, 40]
geoFitTarSet = ["+0", "+1"]
invivoDirSet = [
        "20230703_BY_contrast_trend_upward",
        "20230706_HY_contrast_trend_upward",
        "20230630_EU_contrast_trend_upward"            
        ]
invivo_df_set = {}
for invivoDir in invivoDirSet:
    subject = invivoDir.split("_")[1]
    invivo_df_set[subject] = pd.read_csv(os.path.join(invivoDir, f"{subject}_contrast_trend.csv"))

for subject, invivodata in invivo_df_set.items():
    invivodata_tar = invivodata[(invivodata["SDS [mm]"] >= sdsFitRange[0]) & (invivodata["SDS [mm]"] <= sdsFitRange[-1])]
    invivodata_tar_sds = invivodata_tar["SDS [mm]"]
    invivodata_tar_con = invivodata_tar["Rmax / Rmin  [-]"]
    for geoFitTar in geoFitTarSet:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        for sim_type, simdata in contrast_df_set.items():
            geosim = sim_type.split("_")[1]
            mus = int(sim_type.split("_")[4][:-1])
            mua = int(sim_type.split("_")[6][:-1])
            if (geoFitTar == geosim) & (mus+mua != 200):
                simdata_tar = simdata[(simdata["SDS [mm]"] >= sdsFitRange[0]) & (simdata["SDS [mm]"] <= sdsFitRange[-1])]
                simdata_tar_sds = simdata_tar["SDS [mm]"]
                simdata_tar_con = simdata_tar["Rmax/Rmin [-]"]
                ax[0].plot(simdata_tar_sds, simdata_tar_con-1, "--", label=sim_type)
                ax[1].plot(simdata_tar_sds, (simdata_tar_con-1)/(simdata_tar_con-1).mean(), "--")
        ax[0].plot(invivodata_tar_sds, invivodata_tar_con-1, "-", label=subject)
        ax[1].plot(invivodata_tar_sds, (invivodata_tar_con-1)/(invivodata_tar_con-1).mean(), "-")
        ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[0].set_xlabel('SDS [mm]')
        ax[0].set_ylabel('Rmax/Rmin -1  [-]')
        ax[1].set_xlabel('SDS [mm]')
        ax[1].set_ylabel('Normalized (Rmax/Rmin -1)  [-]')
        ax[0].legend(bbox_to_anchor=(2.25, 1), loc=2, borderaxespad=0. )
        # plt.tight_layout()
        fig.suptitle(f"Comparison of {subject}'s data")
        plt.show()




