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
import utils
import string
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
# plt.rcParams.update({"mathtext.default": "regular"})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 600

outputPath = "/media/md703/Expansion/syu/ijv_2_output/20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99"
sessionIDSet = glob(os.path.join(outputPath, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (float(x.split("_")[-2]), -ord(x.split("_")[1][0])))
sessionIDSet = np.array(sessionIDSet).reshape(5, 2)
sdsObservedRangeSet = [
                        # [0, 53],
                        # [0, 34],
                        [0, 42],
                        # [0, 26]
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
contrastAll = []
for sdsObservedRange in sdsObservedRangeSet: 
    targetSdsSet = np.array(sdsSet[sdsObservedRange[0]: sdsObservedRange[1]])
    
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
            resultPathSet = []            
            resultPathSet += glob(os.path.join(outputPath, sessionID, "post_analysis", "*mua_50%.json"))
            # resultPathSet += glob(os.path.join(outputPath, sessionID, "post_analysis", 
            #                                    "*mua_skin_0%_fat_0%_muscle_0%_ijv_0%_cca_0%.json"))
            resultPathSet += glob(os.path.join(outputPath, sessionID, "post_analysis", 
                                               "*mua_skin_50%_fat_50%_muscle_50%_ijv_50%_cca_50%.json"))
            # resultPathSet += glob(os.path.join(outputPath, sessionID, "post_analysis", 
            #                                    "*mua_skin_100%_fat_100%_muscle_100%_ijv_100%_cca_100%.json"))
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
        labels = [l.get_label() for l in lns]
        ax[0].legend(lns, labels, loc='upper center')
        axtmp_0.set_ylabel("CV [-]")
        axtmp_0.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[0].set_yscale("log")
        ax[0].set_xlabel("SDS [mm]")
        ax[0].set_ylabel("Reflectance Mean [-]")
        ax[0].set_title(f"{'_'.join(sessionID.split('_')[2:])} - reflectance info")
        
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
            contrastAll.append(contrastLocal)
            lns += ax[-1].plot(targetSdsSet, 
                               contrastLocal, 
                               label="Rmax/Rmin", marker=".", linestyle="-")
            lns += axtmp_1.plot(targetSdsSet, contrastCv[idx, sdsObservedRange[0]:sdsObservedRange[1]],
                                label="CV", color=colorset[1], linestyle="--")
            # if sdsObservedRange[-1] <= 26:
            df = np.concatenate((targetSdsSet[:, None], contrastLocal[:, None]), axis=1)
            df = pd.DataFrame(df, columns=["SDS [mm]", "Rmax/Rmin [-]"])
            print(df)
            # df.to_csv(f"contrast_{sessionID[8:]}_mus_50%_mua_{key.split('_')[-1]}.csv",
            #           index=False)
        labels = [l.get_label() for l in lns]
        ax[-1].legend(lns, labels)
        ax[-1].grid()
        axtmp_1.set_ylabel("CV [-]")
        axtmp_1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax[-1].set_xlabel("SDS [mm]")
        ax[-1].set_ylabel("Rmax / Rmin [-]")
        ax[-1].set_title(f"{'_'.join(sessionID.split('_')[2:])} - contrast")
        
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
        with open(os.path.join(sessionID, "model_parameters.json")) as f:
            modelParam = json.load(f)
        depth = modelParam["GeoParam"]["IJV"]["Depth"]
        ijvupperedge2surf.append(depth-modelParam["GeoParam"]["IJV"]["MinorAxisNormal"])
        maxSDS.append((depth, sdsSet[np.argmax(contrastMean[0, :26])]))

print(ijvupperedge2surf)

contrastAll = np.array(contrastAll) - 1
titleSet = [
            "Depth = 8.708 mm", 
            "Depth = 10.88 mm",
            "Depth = 14.5 mm",
            "Depth = 18.12 mm",
            "Depth = 21.74 mm",            
            ]
# %%  plot all contrast in the same plot - raw
fig, ax = plt.subplots(3, 2, figsize=(8, 8))  # (9, 8)
for idx, contrast in enumerate(contrastAll):
    row = idx % 3
    col = idx // 3
    ax[row, col].plot(targetSdsSet, contrast, 
               marker=".", 
               linestyle="-",
               # color=utils.colorFader(c1="cornflowerblue", c2="navy", mix=idx/(contrastAll.shape[0]-1)),
                label="Contrast"
               )
    if idx <= 3:
        mask = targetSdsSet<24.5
        sdsLoc = targetSdsSet[mask]
        contrastLoc = contrast[mask]
        contrastPeak = np.sort(contrastLoc)[-4]
        mask = contrastLoc >= contrastPeak
        sdsLoc = sdsLoc[mask]
        contrastLoc = contrastLoc[mask]
        ax[row, col].plot(sdsLoc, contrastLoc, 
                   marker=".", 
                   linestyle="-",
                    color="red",
                    label="Local Maximum"
                   )
        ax[row, col].legend(edgecolor="black", fontsize="medium")
    ax[row, col].text(0, 1.05, f"({string.ascii_lowercase[idx]})", 
                      transform=ax[row, col].transAxes, 
                      size=13)
    ax[row, col].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax[row, col].set_ylabel("Contrast [-]", size="large")
    ax[row, col].set_title(titleSet[idx], size="large")
    ax[row, col].grid(visible=False)
    ax[row, col].set_xlabel("SDS [mm]", size="large")
# ax[2, 0].set_xlabel("SDS [mm]", size="large")
ax[2, 1].set_xlabel("SDS [mm]", size="large")
# plt.setp(ax[0, 0].get_xticklabels(), visible=False)
# plt.setp(ax[1, 0].get_xticklabels(), visible=False)
# plt.setp(ax[0, 1].get_xticklabels(), visible=False)
# plt.setp(ax[1, 1].get_xticklabels(), visible=False)

#  plot all contrast in the same plot - normalized
for idx, contrast in enumerate(contrastAll):
    contrast -=contrast.min()
    contrast /= contrast.max()
    ax[2, 1].plot(targetSdsSet, contrast, 
             # marker=".", 
             linestyle="-",
             color=utils.colorFader(c1='tab:blue', c2="tab:red", mix=idx/(contrastAll.shape[0]-1)),
             label=f"Depth={titleSet[idx].split(' ')[-2]}mm")
    # if idx <= 3:
    #     argmax = np.argmax(contrast[:22])
    #     if idx == 0:
    #         plt.plot(targetSdsSet[argmax-1:argmax+2], contrast[argmax-1:argmax+2], 
    #                  linewidth=2, 
    #                  linestyle="-",
    #                  color="red",
    #                  label="local peak")
    #     else:
    #         plt.plot(targetSdsSet[argmax-1:argmax+2], contrast[argmax-1:argmax+2], 
    #                  linewidth=2, 
    #                  linestyle="-",
    #                  color="red")
# handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 2, 3, 4, 5, 1]
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
ax[2, 1].grid(visible=False)
ax[2, 1].text(0, 1.05, "(f)", 
                  transform=ax[2, 1].transAxes, 
                  size=13)
ax[2, 1].legend(edgecolor="black", fontsize="small")
ax[2, 1].set_ylabel("Normalized Contrast  [-]", size="large")
ax[2, 1].set_title("Comparison", size="large")
plt.tight_layout()
plt.show()


# plot relationship of depth and optimal-sds
def get_optsds(depth, a, b, c, d):
    optsds = a*depth**4 + b*depth**2 + c*depth + d
    # optsds = a*depth**2 + b*depth**1 + c
    return optsds

ijvupperedge2surf = np.around(ijvupperedge2surf, 3)[:-1]
depthSet = np.array(list(zip(*maxSDS))[0][:-1])
opSds = np.array(list(zip(*maxSDS))[1][:-1])
popt, pcov = curve_fit(get_optsds, depthSet, opSds, maxfev=5000)
plt.plot(depthSet, opSds, "o", label="sim data")
depthSet_fit = np.linspace(depthSet[0], depthSet[-1], 100)
plt.plot(depthSet_fit, get_optsds(depthSet_fit, *popt), label="fit")
plt.grid()
plt.legend()
plt.xlabel("depth  [mm]")
plt.ylabel("optimal sds in sim  [mm]")
plt.title("Relationship of depth and optimal sds based on sim")
plt.show()

# df = pd.DataFrame(np.concatenate((ijvupperedge2surf[:, None], depthSet[:, None], opSds[:, None]), axis=1), columns =["IJV upper edge - skin surface [mm]", "IJV depth [mm]", "all mus mua 50% - OptSDS [mm]"])
# print(df)
# df.to_csv(os.path.join("/".join(os.getcwd().split("/")[:-1]), "sim_contrast_trend.csv"), 
#           index=False)

normalRmax = reflectanceSet["+0_std"]["ijv_col"]["cca_50%"].mean(axis=1)
normalRmin = reflectanceSet["+0_std"]["ijv_dis"]["cca_50%"].mean(axis=1)
normalRmaxdf = pd.DataFrame({"SDS": targetSdsSet, "Reflectance": normalRmax})
normalRmindf = pd.DataFrame({"SDS": targetSdsSet, "Reflectance": normalRmin})
# normalRmaxdf.to_csv("rmax_depth_+0_std_mus_50%_mua_50%.csv", index=False)
# normalRmindf.to_csv("rmin_depth_+0_std_mus_50%_mua_50%.csv", index=False)




