#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:47:40 2022

@author: md703
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import json
import os
from glob import glob
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

outputPath = "/media/md703/Expansion/syu/ijv_2_output/20230715_contrast_invivo_geo_simulation"
subject = "EU"
date = "20230630"
sessionIDSet = glob(os.path.join(outputPath, "*EU*"))  # HY_another
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (
                                float(x.split("_")[-7][:-1]), 
                                -ord(x.split("_")[1][0])))
sessionIDSet = np.array(sessionIDSet).reshape(-1, 2)
sdsObservedRangeSet = [
                        # [0, 53],
                        # [0, 34],
                        # [0, 47],
                        [5, 47]
                       ]
colorset = ["royalblue", "orange"]


# %% PLOT CV and contrast (dis, col in the same plot)
resultPath = glob(os.path.join(outputPath, sessionIDSet[0, 0], "post_analysis", "*"))[0]
with open(resultPath) as f:
    result = json.load(f)
cv = result["MovingAverageGroupingSampleCV"]
sdsSet = []
for key in cv.keys():
    sdsSet.append(float(key[4:]))
sdsSet = np.array(sdsSet)
ijvupperedge2surf = []


maxSDS = []
contrastAll = {}
for sdsObservedRange in sdsObservedRangeSet: 
    simSdsSet = sdsSet[sdsObservedRange[0]: sdsObservedRange[1]]
    
    # cv & contrast
    reflectanceSet = {}    
    for musType in sessionIDSet:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        axtmp_0 = ax[0].twinx()
        axtmp_1 = ax[-1].twinx()
        mus = "_".join(musType[0].split("_")[-2:])
        reflectanceSet[mus] = {}
        lns = []
        for idx, sessionID in enumerate(musType):
            resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*mua_*0%*"))
            resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=True)
            ijvType = "_".join(sessionID.split("_")[:2])
            reflectanceSet[mus][ijvType] = {}
            for resultPath in resultPathSet:
                with open(resultPath) as f:
                    result = json.load(f)
                photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
                muaType = "_".join(resultPath.split("_")[-2:])
                muaType = muaType.split(".")[0]
                reflectanceSet[mus][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))  # [sdsObservedRange[0]: sdsObservedRange[1]]
                reflectance = result["MovingAverageGroupingSampleMean"]
                cv = result["MovingAverageGroupingSampleCV"]
                cv = np.array(list(cv.values()))
                cv /= np.sqrt(10)
                lns += ax[0].plot(simSdsSet, 
                                  list(reflectance.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - ref")
                lns += axtmp_0.plot(simSdsSet, 
                                  cv[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  linestyle=":", 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - cv")
                
        # added these three lines
        title = f"{'_'.join(sessionID.split('_')[2:])}"
        
        labels = [l.get_label() for l in lns]
        ax[0].legend(lns, labels, loc='upper center')
        axtmp_0.set_ylabel("CV [-]")
        axtmp_0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[0].set_yscale("log")
        ax[0].set_xlabel("SDS [mm]")
        ax[0].set_ylabel("Reflectance Mean [-]")
        ax[0].set_title(f"{title} - reflectance info")
        
        # analyze cnr
        rMax = np.array(list(reflectanceSet[mus]["ijv_col"].values()))
        rMin = np.array(list(reflectanceSet[mus]["ijv_dis"].values()))
        contrast = rMax / rMin
        contrastMean = contrast.mean(axis=2)
        contrastStd = contrast.std(ddof=1, axis=2)
        contrastSnr = contrastMean / contrastStd
        contrastCv = contrastStd / contrastMean
        contrastCv /= np.sqrt(10)
        
        lns = []
        for idx, key in enumerate(reflectanceSet[mus]["ijv_dis"].keys()):
            # for sdsIdx, c in enumerate(contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :]):
            #     ax[-1].scatter(np.repeat(simSdsSet[sdsIdx], len(c)), c, marker=".")
            # ax[-1].fill_between(simSdsSet, 
            #                     contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :].min(axis=1),
            #                     contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :].max(axis=1),
            #                     alpha=0.4)
            contrastLocal = contrastMean[idx, sdsObservedRange[0]:sdsObservedRange[1]] - 1
            contrastAll[f"mus_{mus.split('_')[1]}_mua_{key.split('_')[1]}"] = contrastLocal
            lns += ax[-1].plot(simSdsSet, 
                               contrastLocal, 
                               label=f"{key.split('_')[1]}", marker=".", linestyle="-")
            lns += axtmp_1.plot(simSdsSet, contrastCv[idx, sdsObservedRange[0]:sdsObservedRange[1]],
                                # label="CV", 
                                color=lns[-1].get_color(), linestyle="--")
        labels = [l.get_label() for l in lns]
        ax[-1].legend(lns, labels)
        ax[-1].grid()
        axtmp_1.set_ylabel("CV [-]")
        axtmp_1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[-1].set_xlabel("SDS [mm]")
        ax[-1].set_ylabel("Rmax/Rmin - 1  [-]")
        ax[-1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[-1].set_title(f"{title} - contrast")
        
        # show plot of this mus
        plt.tight_layout()
        plt.show()

# %% compare invivo and sim
invivo_df = pd.read_csv(os.path.join(f"/home/md703/syu/ijv_2/{date}_{subject}_contrast_trend_upward", f"{subject}_contrast_trend.csv"))
invivoSdsSet = invivo_df['SDS [mm]'].values
invivoContrast = invivo_df['Rmax / Rmin  [-]'].values - 1
# sim_mask = (simSdsSet > min(invivoSdsSet)) & (simSdsSet < max(invivoSdsSet))
invivo_mask = (invivoSdsSet > min(simSdsSet)) & (invivoSdsSet < max(simSdsSet))

# calculate error
def get_mae(f, y):
    error = f-y
    mae = np.abs(error).mean()
    return mae

error = []
contrastAll_p = {}
for mua, con in contrastAll.items():
    con_p = np.interp(invivoSdsSet[invivo_mask], 
                      simSdsSet, con)
    contrastAll_p[mua] = con_p
    error.append((mua, get_mae(con_p, invivoContrast[invivo_mask])))
error.sort(key=lambda x: x[1])

# plot - compare all
plt.figure(figsize=(5.7, 3))
for mua, con in contrastAll.items():
    # mua
    if mua.split("_")[-1] == "0%":
        color = "red"
    elif mua.split("_")[-1] == "50%":
        color = "orange"
    else:
        color = "brown"
     
    # mus
    if mua.split("_")[1] == "0%":
        marker = "."
    elif mua.split("_")[1] == "50%":
        marker = "*"
    else:
        marker = "v"
    musvalue = mua.split("_")[1][:-1] + "\%"
    print(4-len(mua.split("_")[1]))
    muavalue = mua.split("_")[3][:-1] + "\%"
    plt.plot(simSdsSet, con, "--", color=color, marker=marker, 
             label=f"Sim: ($\mu_s, \mu_a$) = ({musvalue}, {muavalue})", 
             alpha=0.65)
plt.plot(invivoSdsSet, invivoContrast, label=f"Experimental: {subject}",
         linewidth=2)
plt.ylim(-0.07, 0.25)
plt.grid(visible=False)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.legend(edgecolor="black", bbox_to_anchor=(1.01, 1.03), 
           fontsize="small")
plt.xlabel("SDS [mm]")
plt.ylabel("Contrast [-]")
plt.title(f"Comparison - Subject: {subject}")
plt.show()

# contrastAllvalues = np.array(list(contrastAll.values()))
# ub = contrastAllvalues.max(axis=0)
# lb = contrastAllvalues.min(axis=0)
# plt.plot(invivoSdsSet, invivoContrast, label=subject)
# plt.fill_between(simSdsSet, ub, lb, alpha=0.4, label="sim range")
# plt.ylim(-0.07, 0.25)
# plt.grid()
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.legend()
# plt.xlabel("SDS [mm]")
# plt.ylabel("Contrast")
# plt.title(f"Comparison - Subject: {subject}")
# plt.show()

# plot - compare min error and invivo
plt.figure(figsize=(5.8, 3.1))
musvalue = error[0][0].split("_")[1][:-1] + "\%"
muavalue = error[0][0].split("_")[3][:-1] + "\%"
plt.plot(invivoSdsSet[invivo_mask], contrastAll_p[error[0][0]], 
         linestyle="-", marker=".", color="tab:green",
         label=f"Sim: ($\mu_s, \mu_a$) = ({musvalue}, {muavalue})" + f", MAE = {round(error[0][1], 4)*100}\%")
plt.plot(invivoSdsSet[invivo_mask], invivoContrast[invivo_mask], 
         linestyle="-", marker=".", 
         label=f"Experimental: {subject}")
plt.grid(visible=False)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.legend(edgecolor="black", fontsize="small")
plt.xlabel("SDS [mm]")
plt.ylabel("Contrast [-]")
plt.title(f"Fitting result - Subject: {subject}")
plt.show()