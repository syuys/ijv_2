#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:26:48 2022

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import os
import json
from postprocessor import postprocessor
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% parameters
processorSet = {}
sessionIDSet = ["ijv_dis_mus_lb_bloodG_low", "ijv_col_mus_lb_bloodG_low"]
projectRsamples = {}
projectR = {}
projectCV = {}
# for sessionID in sessionIDSet:
#     processorSet[sessionID] = postprocessor(sessionID)


#%% plot reflectance and some ratios
detectorNA = 0.22
fig, ax1 = plt.subplots()
for sessionID in sessionIDSet:
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)
    
    # # re-calculate reflectance with different NA
    # with open(os.path.join(sessionID, "mua.json")) as f:
    #     mua = json.load(f)
    # muaUsed =[mua["1: Air"],
    #           mua["2: PLA"],
    #           mua["3: Prism"],
    #           mua["4: Skin"],
    #           mua["5: Fat"],
    #           mua["6: Muscle"],
    #           mua["7: Muscle or IJV (Perturbed Region)"],
    #           mua["8: IJV"],
    #           mua["9: CCA"]
    #           ]
    # raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = processorSet[sessionID].analyzeReflectance(mua=muaUsed, detectorNA=detectorNA)
    
    simResultPath = os.path.join(config["OutputPath"], sessionID, "post_analysis", "{}_simulation_result.json".format(sessionID))
    with open(simResultPath) as f:
        simResult = json.load(f)
    
    sdsSet = []
    
    for key in simResult["MovingAverageGroupingSampleMean"].keys():
        sdsSet.append(float(key[4:]))
    
    projectRsamples[sessionID] = np.array(list(simResult["MovingAverageGroupingSampleValues"].values()))
    
    reflectanceSet = list(simResult["MovingAverageGroupingSampleMean"].values())
    projectR[sessionID] = np.array(reflectanceSet)
    projectCV[sessionID] = simResult["MovingAverageGroupingSampleCV"]
    
    ax1.plot(sdsSet, reflectanceSet, linestyle="-", marker=".", label=sessionID)


projectRsamples = np.concatenate((projectRsamples["ijv_col_mus_lb_bloodG_low"][None, :, :], 
                                  projectRsamples["ijv_dis_mus_lb_bloodG_low"][None, :, :]
                                  ), 
                                 axis=0)

projectRmeanStd = projectRsamples.mean(axis=0).std(axis=1, ddof=1)
projectRmeanMean = projectRsamples.mean(axis=(0, -1))
projectRmeanCV = projectRmeanStd / projectRmeanMean
z = np.polyfit(sdsSet, projectRmeanCV, 3)
p = np.poly1d(z)
projectRmeanCVfit = p(sdsSet)

Rmax = projectR["ijv_col_mus_lb_bloodG_low"]
RmaxCV = np.array(list(projectCV["ijv_col_mus_lb_bloodG_low"].values()))
Rmin = projectR["ijv_dis_mus_lb_bloodG_low"]
RminCV = np.array(list(projectCV["ijv_dis_mus_lb_bloodG_low"].values()))
Rmean = (Rmax+Rmin) / 2
CVmean = (RmaxCV+RminCV) / 2

z = np.polyfit(sdsSet, CVmean, 3)
p = np.poly1d(z)
CVfit = p(sdsSet)

# ax1.plot(sdsSet, Rmax-Rmin, linestyle="-", marker=".", label="col - dis")
ax1.set_yscale('log')
ax1.legend(loc="upper left")
ax1.set_xlabel("sds [mm]") 
ax1.set_ylabel("reflectance [-]")

contrast = (Rmax-Rmin) / Rmean
cnr = contrast / projectRmeanCVfit
ax2 = ax1.twinx()
ax2.set_ylabel("ΔR / R  [-]")
# ax2.set_ylabel("Ratio [-]")
ax2.plot(sdsSet, contrast, "-*", color="gray")
# ax2.legend(loc="upper right")

plt.show()


# plot Rmax-Rmin
plt.plot(sdsSet, Rmax-Rmin, linestyle="-", marker=".", label="col - dis")
# plt.yscale("log")
plt.xlabel("sds [mm]")
plt.ylabel("col - dis  [-]")
plt.show()


# # plot cnr
# plt.plot(sdsSet, cnr, "-o")
# plt.xlabel("sds [mm]")
# plt.ylabel("contrast / cv_fit  [-]")
# plt.show()


# fit and plot cv
# colorSet = ["tab:blue", "tab:orange"]
for sessionID in sessionIDSet:
    # z = np.polyfit(sdsSet, list(projectCV[sessionID].values()), 3)
    # p = np.poly1d(z)
    plt.plot(sdsSet, projectCV[sessionID].values(),  marker=".", linestyle="-", label=sessionID)
    # plt.plot(sdsSet, p(sdsSet), "--", c=color)
    plt.xlabel("sds [mm]")
    plt.ylabel("CV [-]")
plt.plot(sdsSet, projectRmeanCV,  marker=".", linestyle="-", label="Rmean", color="tab:gray")
plt.plot(sdsSet, projectRmeanCVfit, linestyle="--", label="Rmean_fit", color="tab:gray")
plt.legend()
plt.show()

z = np.polyfit(sdsSet, CVmean, 3)
p = np.poly1d(z)
plt.plot(sdsSet, CVmean, marker=".", linestyle="-", label="average cv")
plt.plot(sdsSet, CVfit, "--", label="cv fit")
plt.legend()
plt.xlabel("sds [mm]")
plt.ylabel("CV [-]")
plt.show()

print(f"Detector NA = {detectorNA}")


    
    
    