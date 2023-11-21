#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 00:33:13 2023

@author: md703
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams["figure.dpi"] = 600

# parameters
tissueSet = [
            # "skin", "fat", "muscle",
            "blood"
             ]
gSet = [
        0.9, 0.9, 0.9, 
        # 0.99
        ]
abfile = "/home/md703/syu/ijv_2/shared_files/model_input_related/optical_properties/ab_bound_of_musp.json"
percentile = 0.5
wltarget = np.array([730, 760, 780, 810, 850])
wl = np.linspace(725, 875, 151)

# main
with open(abfile) as f:
    abBound = json.load(f)
muspBaseWl = abBound["base wl"]
mus = np.empty((len(tissueSet), len(wl)))
mustarget = np.empty((len(tissueSet), len(wltarget)))
for idx, tissue in enumerate(tissueSet):
    bound = abBound[tissue]
    a = bound["a"][0] + (bound["a"][1]-bound["a"][0]) * percentile
    b = bound["b"][0] + (bound["b"][1]-bound["b"][0]) * percentile
    musp = a * (wl/muspBaseWl) ** (-b)
    mus[idx] = musp / (1-gSet[idx])
    musptarget = a * (wltarget/muspBaseWl) ** (-b)
    mustarget[idx] = musptarget / (1-gSet[idx])

# plot
# plt.figure(figsize=(2.8, 2.2))
muslns = []
for idx, tissue in enumerate(tissueSet):
    muslns += plt.plot(wl, mus[idx], 
                       label=tissue.title()
                       )
    wlline = plt.plot(wltarget, mustarget[idx], ".", color="red", 
                      label="730, 760, 780,\n810, 850 nm"
                      )
    # plt.ylim(42, 79)
# plt.yscale("log")
legend1 = plt.legend(wlline, [wlline[0].get_label()], edgecolor="white", 
                     fontsize="small", 
                       # bbox_to_anchor=[0.55, 0.5],
                        bbox_to_anchor=[0.48, 0.75],
                     )
plt.legend(muslns, [ln.get_label() for ln in muslns], edgecolor="black", 
           fontsize="small")
plt.gca().add_artist(legend1)
plt.xlabel("Wavelength [nm]")
plt.ylabel("$\mu_s$ [$cm^{-1}$]")
plt.grid(visible=False)
# plt.title(tissue.title())
plt.show()

# check if calculated mus is the same as the data used in simulation
checkTarSet = np.empty((len(tissueSet), len(wltarget)))
for idx, tissue in enumerate(tissueSet):
    checktissue = tissue.title() if tissue != "blood" else "IJV"
    checkTar = []
    for wl in wltarget:
        with open(os.path.join(f"ijv_dis_op_avg_{wl}nm", "model_parameters.json")) as f:
            modelParam = json.load(f)
        checkTar.append(modelParam["OptParam"][checktissue]["mus"])
    checkTarSet[idx] = np.array(checkTar) * 10  # mm -> cm
print(np.round(mustarget, 5) == np.round(checkTarSet, 5))





