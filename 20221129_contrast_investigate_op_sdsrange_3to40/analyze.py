#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:47:40 2022

@author: md703
"""

import numpy as np
import json
import os
from glob import glob
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

outputPath = "/media/md703/Expansion/syu/ijv_2_output/20221129_contrast_investigate_op_sdsrange_3to40"
sessionIDSet = glob(os.path.join(outputPath, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (int(x.split("_")[-1][:-1]), x.split("_")[1]))
sessionIDSet = np.array(sessionIDSet).reshape(5, 2)


# %% PLOT CV
for musType in sessionIDSet:
    fig, ax = plt.subplots(1, 2, figsize=(11,3))
    for idx, sessionID in enumerate(musType):
        resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
        resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=True)        
        for resultPath in resultPathSet:
            with open(resultPath) as f:
                result = json.load(f)            
            cv = result["MovingAverageGroupingSampleCV"]
            sdsSet = []
            for key in cv.keys():
                sdsSet.append(float(key[4:]))
            label = "_".join(resultPath.split("_")[-2:])
            label = label.split(".")[0]
            ax[idx].plot(sdsSet, cv.values(), label=label)
        photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
        ax[idx].set_xlabel("sds [mm]")
        ax[idx].set_ylabel("cv [-]")
        ax[idx].legend()
        ax[idx].set_title(f"{sessionID} - {photonNum}")
    plt.show()


# %% PLOT REFLECTANCE
# for musType in sessionIDSet:
#     fig, ax = plt.subplots(1, 2, figsize=(11, 3))
#     for idx, sessionID in enumerate(musType):
#         resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
#         resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=True)
#         for resultPath in resultPathSet:
#             with open(resultPath) as f:
#                 result = json.load(f)
#             muaType = "_".join(resultPath.split("_")[-2:])
#             muaType = muaType.split(".")[0]
#             reflectance = result["MovingAverageGroupingSampleMean"]
#             sdsSet = []
#             for key in reflectance.keys():
#                 sdsSet.append(float(key[4:]))            
#             ax[idx].plot(sdsSet, reflectance.values(), label=muaType)
#         photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
#         ax[idx].set_yscale("log")
#         ax[idx].set_xlabel("sds [mm]")
#         ax[idx].set_ylabel("reflectance mean [-]")
#         ax[idx].legend()
#         ax[idx].set_title(f"{sessionID} - {photonNum}")
    
#     # show plot of this mus
#     plt.show()
    

# %% PLOT CNR
# reflectanceSet = {}
# for musType in sessionIDSet:
#     fig, ax = plt.subplots(1, 3, figsize=(16.5, 3))
#     mus = "_".join(musType[0].split("_")[-2:])
#     reflectanceSet[mus] = {}
#     for idx, sessionID in enumerate(musType):
#         resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
#         resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=True)
#         ijvType = "_".join(sessionID.split("_")[:2])
#         reflectanceSet[mus][ijvType] = {}
#         for resultPath in resultPathSet:
#             with open(resultPath) as f:
#                 result = json.load(f)
#             muaType = "_".join(resultPath.split("_")[-2:])
#             muaType = muaType.split(".")[0]
#             reflectanceSet[mus][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
#             reflectance = result["MovingAverageGroupingSampleMean"]
#             sdsSet = []
#             for key in reflectance.keys():
#                 sdsSet.append(float(key[4:]))            
#             ax[idx].plot(sdsSet, reflectance.values(), label=muaType)
#         photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
#         ax[idx].set_yscale("log")
#         ax[idx].set_xlabel("sds [mm]")
#         ax[idx].set_ylabel("reflectance mean [-]")
#         ax[idx].legend()
#         ax[idx].set_title(f"{sessionID} - {photonNum}")
#     # analyze cnr
#     rMax = np.array(list(reflectanceSet[mus]["ijv_col"].values()))
#     rMin = np.array(list(reflectanceSet[mus]["ijv_dis"].values()))
#     rMaxMean = rMax.mean(axis=-1)
#     rMinMean = rMin.mean(axis=-1)
#     contrast = rMaxMean / rMinMean
#     # rMean = (rMax + rMin) / 2
#     # rMeanStd = rMean.std(axis=-1, ddof=1)
#     # rMeanMean = rMean.mean(axis=-1)
#     # rMeanCV = rMeanStd / rMeanMean
#     # cnr = ((rMaxMean - rMinMean) / rMeanMean) / rMeanCV
    
#     for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
#         ax[-1].plot(sdsSet, contrast[idx], label=key, marker=".", linestyle="-")
#     ax[-1].set_xlabel("sds [mm]")
#     ax[-1].set_ylabel("Rmax / Rmin [-]")
#     ax[-1].legend()
#     ax[-1].set_title(f"{sessionID} - {photonNum}")
    
#     # show plot of this mus
#     plt.show()


# %% PLOT CNR - dis, col in the same plot
reflectanceSet = {}
colorset = [["cornflowerblue", "sandybrown"],
            ["royalblue", "orange"],
            ["blue", "darkorange"],
            ["mediumblue", "chocolate"],
            ["darkblue", "saddlebrown"]]
for musType in sessionIDSet:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    axtmp = ax[0].twinx()
    mus = "_".join(musType[0].split("_")[-2:])
    reflectanceSet[mus] = {}
    lns = []
    slopeSet = []
    for idx2, sessionID in enumerate(musType):
        resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
        resultPathSet.sort(key=lambda x: int(x.split("_")[-1].split("%")[0]), reverse=True)
        ijvType = "_".join(sessionID.split("_")[:2])
        reflectanceSet[mus][ijvType] = {}
        for idx1, resultPath in enumerate(resultPathSet):
            with open(resultPath) as f:
                result = json.load(f)
            photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
            muaType = "_".join(resultPath.split("_")[-2:])
            muaType = muaType.split(".")[0]
            reflectanceSet[mus][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
            reflectance = result["MovingAverageGroupingSampleMean"]
            sdsSet = []
            for key in reflectance.keys():
                sdsSet.append(float(key[4:]))
            refDiff = np.diff(np.log10(list(reflectance.values())))
            sdsDiff = np.diff(sdsSet)
            slope = refDiff / sdsDiff
            slopeSet.append(slope)
            slope_x = np.array(sdsSet)[:-1] + sdsDiff
            lns += ax[0].plot(sdsSet, reflectance.values(), color=colorset[idx1][idx2], 
                              label=f"{sessionID.split('_')[1]} - {muaType}")
            # lns += axtmp.plot(slope_x, slope, linestyle=":", color=colorset[idx2], 
            #                   label=f"{sessionID.split('_')[1]} - slope")
            # ax[0].plot(slope_x, slope, marker=".", linestyle="--", color=colorset[idx2], 
            #            label=f"{sessionID.split('_')[1]} - {photonNum}")
            
    # added these three lines
    labels = [l.get_label() for l in lns]
    # ax[0].legend(lns, labels, loc="upper center")
    ax[0].legend(lns, labels, fontsize=9, loc="lower left")
    # axtmp.set_ylabel("slope [-]")
        
    ax[0].set_yscale("log")
    ax[0].set_xlabel("sds [mm]")
    ax[0].set_ylabel("reflectance mean [-]")
    # ax[0].legend()
    ax[0].set_title(f"{'_'.join(sessionID.split('_')[2:])} - reflectance")
    
    # analyze cnr
    rMax = np.array(list(reflectanceSet[mus]["ijv_col"].values()))
    rMin = np.array(list(reflectanceSet[mus]["ijv_dis"].values()))
    rMaxMean = rMax.mean(axis=-1)
    rMinMean = rMin.mean(axis=-1)
    contrast = rMaxMean / rMinMean
    # rMean = (rMax + rMin) / 2
    # rMeanStd = rMean.std(axis=-1, ddof=1)
    # rMeanMean = rMean.mean(axis=-1)
    # rMeanCV = rMeanStd / rMeanMean
    # cnr = ((rMaxMean - rMinMean) / rMeanMean) / rMeanCV
    
    for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
        ax[-1].plot(sdsSet, contrast[idx], label=key, marker=".", linestyle="-")
    ax[-1].set_xlabel("sds [mm]")
    ax[-1].set_ylabel("Rmax / Rmin [-]")
    ax[-1].legend()
    ax[-1].set_title(f"{'_'.join(sessionID.split('_')[2:])} - contrast")
    
    # show plot of this mus
    plt.tight_layout()
    plt.show()
    
    # # show slope solely
    # for slope in slopeSet:
    #     plt.plot(slope_x, slope)
    # plt.show()