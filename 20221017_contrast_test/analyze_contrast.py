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
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


sessionIDSet = ["ijv_dis_mus_lb_bloodG_low", "ijv_col_mus_lb_bloodG_low"]
projectR = {}
projectCV = {}
fig, ax1 = plt.subplots()

for sessionID in sessionIDSet:
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)
    simResultPath = os.path.join(config["OutputPath"], sessionID, "post_analysis", "{}_simulation_result.json".format(sessionID))
    with open(simResultPath) as f:
        simResult = json.load(f)
    
    sdsSet = []
    
    for key in simResult["MovingAverageGroupingSampleMean"].keys():
        sdsSet.append(float(key[4:]))
    reflectanceSet = list(simResult["MovingAverageGroupingSampleMean"].values())
    projectR[sessionID] = np.array(reflectanceSet)
    projectCV[sessionID] = simResult["MovingAverageGroupingSampleCV"]
    
    ax1.plot(sdsSet, reflectanceSet, "-o", label=sessionID)
    ax1.legend()

ax1.set_xlabel("sds [mm]") 
ax1.set_ylabel("reflectance [-]")

# contrast = Rmax/Rmin (col/dis)
Rmax = projectR["ijv_col_mus_lb_bloodG_low"]
Rmin = projectR["ijv_dis_mus_lb_bloodG_low"]
# contrast = Rmax / Rmin
contrast = (Rmax-Rmin) / ((Rmax+Rmin)/2)
ax2 = ax1.twinx()
ax2.set_ylabel("ΔR / R  [-]")
ax2.plot(sdsSet, contrast, "-*", color="gray")

plt.show()

for sessionID in sessionIDSet:
    plt.plot(sdsSet, projectCV[sessionID].values(), "-o", label=sessionID)
    plt.xlabel("sds [mm]")
    plt.ylabel("CV [-]")
plt.legend()
plt.show()





    
    
    