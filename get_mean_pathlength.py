#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 18:12:48 2023

@author: md703
"""

from postprocess import getMeanPathlength
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])

projectID = "SCS"
sessionID = "run_1"
muaID = "mua_test.json"

simrespath = os.path.join("/home/md703/syu/ijv_2_output", projectID, sessionID, 
                          "post_analysis", 
                          f"{sessionID}_simulation_result.json")
with open(simrespath) as f:
    tmp = json.load(f)
cv = tmp["MovingAverageGroupingSampleCV"]
sdsSet = []
for key in cv.keys():
    sdsSet.append(float(key[4:]))
sdsSet = np.array(sdsSet)

mua = []
with open(os.path.join(projectID, sessionID, muaID)) as f:
    tmp = json.load(f)
mua.append([tmp["1: Air"],
            tmp["2: PLA"],
            tmp["3: Prism"],
            tmp["4: Skin"],
            tmp["5: Fat"],
            tmp["6: Muscle"],
            tmp["7: Muscle or IJV (Perturbed Region)"],
            tmp["8: IJV"],
            tmp["9: CCA"]
            ])
mua = np.array(mua).T

meanPathlength, movingAverageMeanPathlength = getMeanPathlength(projectID, sessionID, mua)

#%%
avg = movingAverageMeanPathlength.mean(axis=0)
# plt.plot(avg[:, 5])
# plt.show()
plt.plot(avg[1, :], label="10")
# plt.show()
plt.plot(avg[12, :], label="20")
plt.legend()
plt.show()
