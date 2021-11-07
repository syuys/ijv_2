#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:25:55 2021

@author: md703
"""

from mcx_ultrasound_model import MCX
import postprocess
import os
import json
from glob import glob
import jdata as jd
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

#%% session name
sessionID = "test_bc"


# %% run with different size
# parameters
sds = 30
# ijvDepth = 20
xFactorSet = np.linspace(5, 5, num=1)
# yFactor = 2.5
# zFactor = 3

reflectanceMeanSet = {}
detectedPhotonRaioSet = {}
for xFactor in xFactorSet:
    volDim = [int(sds*xFactor*2),
              120,
              95
              ]
    print("volDim:", volDim)
    
    # initialize and run
    simulator = MCX(sessionID)
    simulator.replay(volDim=volDim)
    
    # analyze reflectance
    raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = postprocess.analyzeReflectance(sessionID, wl=900)
    
    reflectanceMeanSet["{}".format(xFactor)] = reflectanceMean.reshape(-1, 3, 2).mean(axis=-1)
    
    # observe detected photons
    detectedPhotonRaio = []
    detOutputPathSet = glob(os.path.join(sessionID, "output", "mcx_output", "*.jdat"))  # about paths of detected photon data
    detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
    # detOutputPathSet = detOutputPathSet[:2]
    for detOutputPath in detOutputPathSet:
        info = jd.load(detOutputPath)["MCXData"]["Info"]
        detectedPhotonRaio.append(info["DetectedPhoton"])
    detectedPhotonRaioSet["{}".format(xFactor)] = detectedPhotonRaio
    


# %% analyze reflectance
for key, value in reflectanceMeanSet.items():
    plt.plot(value[:, 1], label=key)
plt.legend()
plt.title("each scaling reflectance")
plt.show()

baseReflectance = reflectanceMeanSet["5.0"]
for key, value in reflectanceMeanSet.items():
    plt.plot((value/baseReflectance)[:, 1], label="{} / 5".format(key))
plt.legend()
plt.title("reflectance difference")
plt.show()

for key, value in detectedPhotonRaioSet.items():
    plt.plot(value, label=key)
plt.legend()
plt.title("detected photons difference")
plt.show()


# %% analyze


