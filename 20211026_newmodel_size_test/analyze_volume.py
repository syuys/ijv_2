#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:25:55 2021

@author: md703
"""

from mcx_ultrasound_model import MCX
import postprocess
import json
from copy import deepcopy
import jdata as jd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% define function
def ceilEven(x):
    if x % 2 == 1:
        x = x + 1
    return x

#%% parameters
sessionID = "test_bc"
with open("test_bc/config.json") as f:
    config = json.load(f)
with open(config["ModelParametersPath"]) as f:
    modelParameters = json.load(f)
baselineDim = [modelParameters["ModelSize"]["XSize"],
               modelParameters["ModelSize"]["YSize"],
               modelParameters["ModelSize"]["ZSize"]
               ]
detectorHolderEdgeLength = 31


# %% run baseline
simulator = MCX(sessionID)
simulator.replay(volDim=baselineDim)
# retreive output and save reflectance
outputPathSet = glob(os.path.join(sessionID, "output", "mcx_output", "*.jdat"))
baselineReflectance = postprocess.getReflectance(1.51, 1.51, 0.22, 18, outputPathSet, config["PhotonNum"])
baselineReflectance = baselineReflectance.reshape(baselineReflectance.shape[0], -1, 3, 2).mean(axis=(0, -1))[-1, 1]


# %% run replay for determining best model size
# define initial factor, error threshold and some containers
factorSet = [1, 1, 1]  # x, y, z
pivotSet = deepcopy(factorSet)
pivotVariationSet = [[], [], []]  # 2d array
decentRate = 0.9
# detpNumVariationSet = [[], [], []]
errorThold = np.linspace(1e-3/3, 1e-3, num=3)
errorVariationSet = [[], [], []]  # 2d array
reflectanceVariationSet = [[], [], []]
# determine best model size
for idx in range(len(factorSet)):
    while True:
        # save current pivot
        pivotVariationSet[idx].append(pivotSet[idx])
        # initialize and run
        print("\n\nmcxInput[Domain][Dim]:", [ceilEven(int(baselineDim[0]*pivotSet[0])), 
                                             ceilEven(int(baselineDim[1]*pivotSet[1])), 
                                             ceilEven(int(baselineDim[2]*pivotSet[2]))
                                             ])
        simulator = MCX(sessionID)
        simulator.replay(volDim=[ceilEven(int(baselineDim[0]*pivotSet[0])), 
                                 ceilEven(int(baselineDim[1]*pivotSet[1])), 
                                 ceilEven(int(baselineDim[2]*pivotSet[2]))
                                 ])
        # retreive output and save reflectance
        # detpNumVariationSet[idx].append(jd.load("test_bc/output/mcx_output/test_bc_900nm_0_detp.jdat")["MCXData"]["Info"]["DetectedPhoton"])
        outputPathSet = glob(os.path.join(sessionID, "output", "mcx_output", "*.jdat"))
        reflectance = postprocess.getReflectance(1.51, 1.51, 0.22, 18, outputPathSet, config["PhotonNum"])
        reflectance = reflectance.reshape(reflectance.shape[0], -1, 3, 2).mean(axis=(0, -1))[-1, 1]
        reflectanceVariationSet[idx].append(reflectance)
        # save current error
        error = (reflectance-baselineReflectance)/baselineReflectance
        errorVariationSet[idx].append(error)
        # judge error. update pivot and factorSet[idx]
        if error > 1e-3:
            raise Exception("reflectance is larger than baseline reflectance !!")
        if abs(error) < errorThold[idx]:
            factorSet[idx] = pivotSet[idx]
            pivotSet[idx] = pivotSet[idx] * decentRate
        else:
            pivotSet[idx] = (factorSet[idx]+pivotSet[idx])/2
        # check condition
        cond1 = ceilEven(int(baselineDim[idx]*pivotSet[idx])) == ceilEven(int(baselineDim[idx]*factorSet[idx]))
        cond2 = ceilEven(int(baselineDim[0]*pivotSet[0]))//2 - detectorHolderEdgeLength < 0
        cond3 = ceilEven(int(baselineDim[1]*pivotSet[1]))//2 <= 16  # consider source holder width + cca edge
        cond4 = ceilEven(int(baselineDim[2]*pivotSet[2])) <= 28  # ijv depth + holder height
        if cond1 or cond2 or cond3 or cond4:
            if cond2:
                pivotSet[0] = pivotSet[0] / decentRate
            if cond3:
                pivotSet[1] = pivotSet[1] / decentRate
            if cond4:
                pivotSet[2] = pivotSet[2] / decentRate
            break;

## visualization
# plot pivot variation
for idx, pivotVariation in enumerate(pivotVariationSet):
    plt.plot(pivotVariation, "-o", label=f"dim_{idx}")
plt.legend()
plt.title("pivot variation")
plt.show()
# plot reflectance variation
for idx, reflectanceVariation in enumerate(reflectanceVariationSet):
    plt.plot(reflectanceVariation, "-o", label=f"dim_{idx}")
plt.legend()
plt.title("reflectance variation")
plt.show()
# plot error variation
for idx, errorVariation in enumerate(errorVariationSet):
    plt.plot(errorVariation, "-o")
    plt.xticks(np.arange(len(errorVariation)), (np.array(pivotVariationSet[idx])*baselineDim[idx]).astype(int).astype(str))
    plt.title(f"error variation, dim_{idx}, sds={detectorHolderEdgeLength}")
    plt.show()
# plot detected photon num variation
# for idx, detpNumVariation in enumerate(detpNumVariationSet):
#     plt.plot(detpNumVariation, "-o")
#     plt.xticks(np.arange(len(detpNumVariation)), (np.array(pivotVariationSet[idx])*baselineDim[idx]).astype(int).astype(str))
#     plt.title(f"detected photon number variation, dim_{idx}, sds={detectorHolderEdgeLength}")
#     plt.show()


# %% save final determined size to replay.json and run and check final error
simulator = MCX(sessionID)
simulator.replay(volDim=[ceilEven(int(baselineDim[0]*factorSet[0])), 
                         ceilEven(int(baselineDim[1]*factorSet[1])), 
                         ceilEven(int(baselineDim[2]*factorSet[2]))
                         ])
reflectance = postprocess.getReflectance(1.51, 1.51, 0.22, 18, ["test_bc/output/mcx_output/test_bc_900nm_0_detp.jdat"], config["PhotonNum"]).reshape(-1, 3, 2).mean(axis=2)[-1, 1]
error = (reflectance-baselineReflectance)/baselineReflectance
print("\nFinal error:", error)
print("Final factor set:", np.round(np.array(factorSet), 3))
print("Final dim:", [ceilEven(int(baselineDim[0]*factorSet[0])), ceilEven(int(baselineDim[1]*factorSet[1])), ceilEven(int(baselineDim[2]*factorSet[2]))])
