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
import utils
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

outputPath = "/media/md703/Expansion/syu/ijv_2_output/20230819_contrast_investigate_ijvpulserange_sdsrange_5to45_g99"
sessionIDSet = glob(os.path.join(outputPath, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (float(x.split("_")[-1][0:-1]), 
                                 -ord(x.split("_")[1][0])))
sessionIDSet = np.array(sessionIDSet).reshape(-1, 2)
sdsObservedRangeSet = [
                        # [0, 53],
                        # [0, 34],
                        [0, 42]
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
ijvupperedge2surf = []


maxSDS = []
contrastAll = {}
for sdsObservedRange in sdsObservedRangeSet: 
    targetSdsSet = sdsSet[sdsObservedRange[0]: sdsObservedRange[1]]
    
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
            resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
            resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=True)
            ijvType = "_".join(sessionID.split("_")[:2])
            reflectanceSet[mus][ijvType] = {}
            for resultPath in resultPathSet:
                with open(resultPath) as f:
                    result = json.load(f)
                photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
                muaType = "_".join(resultPath.split("_")[-2:])
                muaType = muaType.split(".")[0]
                reflectanceSet[mus][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
                reflectance = result["MovingAverageGroupingSampleMean"]
                cv = result["MovingAverageGroupingSampleCV"]
                cv = np.array(list(cv.values()))
                cv /= np.sqrt(10)
                lns += ax[0].plot(targetSdsSet, 
                                  list(reflectance.values())[sdsObservedRange[0]: sdsObservedRange[1]], 
                                  color=colorset[idx], 
                                  label=f"{sessionID.split('_')[1]} - ref")
                lns += axtmp_0.plot(targetSdsSet, 
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
        # rMaxMean = rMax.mean(axis=-1)
        # rMinMean = rMin.mean(axis=-1)
        # contrast = rMaxMean / rMinMean
        # rMean = (rMax + rMin) / 2
        # rMeanStd = rMean.std(axis=-1, ddof=1)
        # rMeanMean = rMean.mean(axis=-1)
        # rMeanCV = rMeanStd / rMeanMean
        # cnr = ((rMaxMean - rMinMean) / rMeanMean) / rMeanCV
        
        # do t test
        # tSet = []
        # for idx in range(1, len(targetSdsSet)):
        #     r1 = contrast[0, idx-1, :]
        #     r2 = contrast[0, idx, :]
        #     t = stats.ttest_ind(r1, r2, equal_var=False)
        #     tSet.append(t[1])
        
        lns = []
        for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
            # for sdsIdx, c in enumerate(contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :]):
            #     ax[-1].scatter(np.repeat(targetSdsSet[sdsIdx], len(c)), c, marker=".")
            # ax[-1].fill_between(targetSdsSet, 
            #                     contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :].min(axis=1),
            #                     contrast[idx, sdsObservedRange[0]:sdsObservedRange[1], :].max(axis=1),
            #                     alpha=0.4)
            contrastLocal = contrastMean[idx, sdsObservedRange[0]:sdsObservedRange[1]]
            contrastAll[title] = contrastLocal
            lns += ax[-1].plot(targetSdsSet, 
                               contrastLocal, 
                               label="Rmax/Rmin", marker=".", linestyle="-")
            lns += axtmp_1.plot(targetSdsSet, contrastCv[idx, sdsObservedRange[0]:sdsObservedRange[1]],
                                label="CV", color=colorset[1], linestyle="--")
        labels = [l.get_label() for l in lns]
        ax[-1].legend(lns, labels)
        ax[-1].grid()
        axtmp_1.set_ylabel("CV [-]")
        axtmp_1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[-1].set_xlabel("SDS [mm]")
        ax[-1].set_ylabel("Rmax / Rmin [-]")
        ax[-1].set_title(f"{title} - contrast")
        
        # show plot of this mus
        plt.tight_layout()
        plt.show()
        
        # # show variation of t test result
        # plt.plot(targetSdsSet[1:], tSet, "-o")
        # plt.xlabel("sds [mm]")
        # plt.ylabel("p value [mm")
        # plt.title(f"{'_'.join(sessionID.split('_')[2:])} - p value variation")
        # plt.show()
        
        
        # record max sds
        # with open(os.path.join(sessionID, "model_parameters.json")) as f:
        #     modelParam = json.load(f)
        # depth = modelParam["GeoParam"]["IJV"]["Depth"]
        # ijvupperedge2surf.append(depth-modelParam["GeoParam"]["IJV"]["MinorAxisNormal"])
        # maxSDS.append((depth, sdsSet[np.argmax(contrastMean[0, :26])]))

# %%
baseline = "/home/md703/syu/ijv_2/20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99/contrast_depth_+0_std_mus_50%_mua_50%.csv"
baselinedf = pd.read_csv(baseline)
# plt.plot(targetSdsSet, contrast-1, label=sessionID)
# for sessionID, contrast in list(contrastAll.items())[:2]:

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.5))
ax[0].plot(targetSdsSet, contrastAll['pulse_50%']-1, label="Pulse: 50\%")
ax[0].plot(baselinedf["SDS [mm]"][sdsObservedRange[0]: sdsObservedRange[1]], 
         baselinedf["Rmax/Rmin [-]"][sdsObservedRange[0]: sdsObservedRange[1]]-1, 
         label="Pulse: 100\% (Average)")
ax[0].plot(targetSdsSet, contrastAll['pulse_150%']-1, label="Pulse: 150\%")
# plt.plot(targetSdsSet, contrast-1, label=sessionID)
# ax[0].legend(edgecolor="black", fontsize="small")
ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax[0].set_xlabel("SDS [mm]")
ax[0].set_ylabel("Contrast [-]")
# plt.title("Comparison - Minor")
ax[0].grid(visible=False)
# plt.show()

# for sessionID, contrast in list(contrastAll.items())[:2]:
#     tmp = contrast.copy()
#     tmp -= 1
#     tmp -= tmp.min()
#     tmp /= tmp.max()
#     plt.plot(targetSdsSet, tmp, label=sessionID)
# tmp = baselinedf["Rmax/Rmin [-]"][sdsObservedRange[0]: sdsObservedRange[1]]-1
# tmp -= tmp.min()
# tmp /= tmp.max()
# plt.plot(baselinedf["SDS [mm]"][sdsObservedRange[0]: sdsObservedRange[1]], 
#          tmp, label="Average")
ax[1].plot(targetSdsSet, utils.normalize(contrastAll['pulse_50%']-1), label="Pulse: 50\%")
ax[1].plot(baselinedf["SDS [mm]"][sdsObservedRange[0]: sdsObservedRange[1]], 
         utils.normalize(baselinedf["Rmax/Rmin [-]"][sdsObservedRange[0]: sdsObservedRange[1]]-1), 
         label="Pulse: 100\% (Average)")
ax[1].plot(targetSdsSet, utils.normalize(contrastAll['pulse_150%']-1), label="Pulse: 150\%")
# ax[1].legend(edgecolor="black", fontsize="small",
#              bbox_to_anchor=(1.01, 1.03))
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, edgecolor="black", 
           loc='lower center', bbox_to_anchor=(0.5, -0.07),
           ncol=3, fontsize="small")
ax[1].set_xlabel("SDS [mm]")
ax[1].set_ylabel("Normalized Contrast [-]")
# plt.title("Comparison - Minor")
ax[1].grid(visible=False)
plt.tight_layout()
plt.show()

