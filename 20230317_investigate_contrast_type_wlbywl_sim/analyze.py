#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:47:40 2022

@author: md703
"""

import numpy as np
import utils
import json
import os
from glob import glob
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

projectID = "20230317_investigate_contrast_type_wlbywl_sim"
outputPath = f"/media/md703/Expansion/syu/ijv_2_output/{projectID}"
sessionIDSet = glob(os.path.join(outputPath, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (int(x[-5:-2]), -ord(x[4])))
sessionIDSet = np.array(sessionIDSet).reshape(-1, 2)


# %% PLOT CV
for musType in sessionIDSet:
    fig, ax = plt.subplots(1, 2, figsize=(11,3))
    for idx, sessionID in enumerate(musType):
        resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
        resultPathSet.sort(key=lambda x: int(x.split(".")[-2][-5:-2]))
        for resultPath in resultPathSet:
            with open(resultPath) as f:
                result = json.load(f)            
            cv = result["MovingAverageGroupingSampleCV"]
            sdsSet = []
            for key in cv.keys():
                sdsSet.append(float(key[4:]))
            muaType = "_".join(resultPath.split("_")[-2:])
            muaType = muaType.split(".")[0]
            ax[idx].plot(sdsSet, cv.values(), label=muaType)
        photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
        ax[idx].set_xlabel("sds [mm]")
        ax[idx].set_ylabel("cv [-]")
        ax[idx].legend()
        ax[idx].set_title(f"{sessionID} - {photonNum}")
    plt.show()


# %% PLOT REFLECTANCE
for musType in sessionIDSet:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    for idx, sessionID in enumerate(musType):
        resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
        resultPathSet.sort(key=lambda x: int(x.split(".")[-2][-5:-2]))
        for resultPath in resultPathSet:
            with open(resultPath) as f:
                result = json.load(f)
            reflectance = result["MovingAverageGroupingSampleMean"]
            sdsSet = []
            for key in reflectance.keys():
                sdsSet.append(float(key[4:])) 
            muaType = "_".join(resultPath.split("_")[-2:])
            muaType = muaType.split(".")[0]                       
            ax[idx].plot(sdsSet, reflectance.values(), label=muaType)
        photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
        ax[idx].set_yscale("log")
        ax[idx].set_xlabel("sds [mm]")
        ax[idx].set_ylabel("reflectance mean [-]")
        ax[idx].legend()
        ax[idx].set_title(f"{sessionID} - {photonNum}")
    
    # show plot of this mus
    plt.show()
    

# %% PLOT CNR
reflectanceSet = {}
for musType in sessionIDSet:
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 3))
    mus = "_".join(musType[0].split("_")[-2:])
    reflectanceSet[mus] = {}
    for idx, sessionID in enumerate(musType):
        resultPathSet = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))
        resultPathSet.sort(key=lambda x: int(x.split(".")[-2][-5:-2]))
        ijvType = "_".join(sessionID.split("_")[:2])
        reflectanceSet[mus][ijvType] = {}
        for resultPath in resultPathSet:
            with open(resultPath) as f:
                result = json.load(f)
            muaType = "_".join(resultPath.split("_")[-2:])
            muaType = muaType.split(".")[0]
            reflectanceSet[mus][ijvType][muaType] = np.array(list(result["MovingAverageGroupingSampleValues"].values()))
            reflectance = result["MovingAverageGroupingSampleMean"]
            sdsSet = []
            for key in reflectance.keys():
                sdsSet.append(float(key[4:]))            
            ax[idx].plot(sdsSet, reflectance.values(), label=muaType)
        photonNum = "{:.2e}".format(float(result["PhotonNum"]["GroupingSample"]))
        ax[idx].set_yscale("log")
        ax[idx].set_xlabel("sds [mm]")
        ax[idx].set_ylabel("reflectance mean [-]")
        ax[idx].legend()
        ax[idx].set_title(f"{sessionID} - {photonNum}")
    # analyze cnr
    rMax = np.array(list(reflectanceSet[mus]["ijv_col"].values()))
    rMin = np.array(list(reflectanceSet[mus]["ijv_dis"].values()))
    rMaxMean = rMax.mean(axis=-1)
    rMinMean = rMin.mean(axis=-1)
    # rMean = (rMax + rMin) / 2
    # rMeanStd = rMean.std(axis=-1, ddof=1)
    # rMeanMean = rMean.mean(axis=-1)
    # rMeanCV = rMeanStd / rMeanMean
    # cnr = ((rMaxMean - rMinMean) / rMeanMean) / rMeanCV
    metricType = "rmax / rmin"
    metric = utils.get_rmax_by_rmin(rMaxMean, rMinMean)
    print(f"longest sds metric: {metric.ravel()[-1]}")
    
    for idx, key in enumerate(reflectanceSet[mus]["ijv_col"].keys()):
        ax[-1].plot(sdsSet, metric[idx], label=key, marker=".", linestyle="-")
    ax[-1].set_xlabel("sds [mm]")
    ax[-1].set_ylabel(f"{metricType} [-]")
    ax[-1].legend()
    ax[-1].set_title(f"{sessionID} - {photonNum}")
    
    # show plot of this mus
    plt.show()