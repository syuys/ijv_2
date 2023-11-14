#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:16:01 2022

@author: md703
"""

import json
import os
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% main
phantomSet = "CHIK"
wlSet = [730, 760, 810, 850]
root = "/home/md703/Desktop/ijv_2_output/20220308_phantom_calibration"
sds = "sds_20.38"
cvThreshold = 0.1581

for phantom in phantomSet:
    spectrum = []
    cvSet = []
    for wl in wlSet:
        sessionID = f"{phantom}_{wl}"
        with open(os.path.join(root, sessionID, "post_analysis", f"{sessionID}_simulation_result.json")) as f:
            result = json.load(f)
        spectrum.append(result["MovingAverageGroupingSampleMean"][sds])
        cvSet.append(result["MovingAverageGroupingSampleCV"][sds])
    
    plt.subplot(1, 2, 1)
    plt.plot(wlSet, spectrum, "-o")
    plt.xlabel("wl")
    plt.ylabel("reflectance")
    
    plt.subplot(1, 2, 2)
    plt.plot(wlSet, cvSet, "-o")
    plt.axhline(y=cvThreshold, color="r", linestyle="-")
    plt.xlabel("wl")
    plt.ylabel("cv")
    
    plt.suptitle(f"phantom {phantom}")
    
    plt.tight_layout()
    plt.show()
    
    