#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:59:03 2023

@author: md703
"""

import numpy as np
from scipy import stats
import json
import os
from glob import glob
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% parameters
projectID = "20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99"
sdsObservedRangeSet = [
                        # [0, 53],
                        [0, 34]
                       ]
colorset = ["royalblue", "orange"]


# %% main

# initialize
outputPath = f"/media/md703/Expansion/syu/ijv_2_output/{projectID}"
sessionIDSet = glob(os.path.join(outputPath, "*"))
sessionIDSet = [sessionID.split("/")[-1] for sessionID in sessionIDSet]
sessionIDSet.sort(key=lambda x: (float(x.split("_")[-2]), -ord(x.split("_")[1][0])))
sessionIDSet = np.array(sessionIDSet).reshape(5, 2)
resultPath = glob(os.path.join(outputPath, sessionIDSet[0, 0], "post_analysis", "*"))[0]
with open(resultPath) as f:
    result = json.load(f)
sdsSet = []
for key in result["MovingAverageGroupingSampleCV"].keys():
    sdsSet.append(float(key[4:]))


# analyze
for sdsObservedRange in sdsObservedRangeSet:
    targetSdsSet = sdsSet[sdsObservedRange[0]: sdsObservedRange[1]]
    for musType in sessionIDSet:
        mus = "_".join(musType[0].split("_")[-3:])
        reflectanceSet = []
        for idx, sessionID in enumerate(musType):
            resultPath = glob(os.path.join(outputPath, sessionID, "post_analysis", "*"))[0]
            with open(resultPath) as f:
                result = json.load(f)
            values = np.array(list(result["MovingAverageGroupingSampleValues"].values()))[sdsObservedRange[0]: sdsObservedRange[1]]
            reflectanceSet.append(values)
        
        # do t test
        reflectanceSet = np.array(reflectanceSet)
        t = stats.ttest_ind(reflectanceSet[0], 
                            reflectanceSet[1], 
                            equal_var=False, axis=1)
        plt.plot(targetSdsSet, t[1], "-o")
        plt.xlabel("sds [mm]")
        plt.ylabel("p value [-]")
        plt.title(mus)
        plt.show()











