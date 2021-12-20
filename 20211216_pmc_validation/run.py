#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:17:21 2021

@author: md703
"""

from mcx_ultrasound_opsbased import MCX
import postprocess
from glob import glob
import json
import os

# %% Setting

sessionID = "large_ijv_mus_baseline"
runningNum = False  # (Integer or False)
cvThreshold = 0.1581
repeatTimes = 10
muaPath = "mua.json"


# %% Do simulation
# initialize
simulator = MCX(sessionID)
with open(os.path.join(sessionID, "config.json")) as f:
    config = json.load(f)
simulationResultPath = os.path.join(config["OutputPath"], sessionID, "post_analysis", "{}_simulation_result.json".format(sessionID))
with open(simulationResultPath) as f:
    simulationResult = json.load(f)
existedOutputNum = simulationResult["RawSampleNum"]
# run forward mcx
if runningNum:    
    for idx in range(existedOutputNum, existedOutputNum+runningNum):
        # run
        simulator.run(idx)
        # save progress        
        simulationResult["RawSampleNum"] = idx+1
        with open(simulationResultPath, "w") as f:
            json.dump(simulationResult, f, indent=4)
else:
    reflectanceCV = simulationResult["GroupingSampleCV"].values()
    while(max(reflectanceCV) > cvThreshold):
        with open(simulationResultPath) as f:
            simulationResult = json.load(f)
        needAddOutputNum = repeatTimes - existedOutputNum % repeatTimes
        for idx in range(existedOutputNum, existedOutputNum+needAddOutputNum):
            # run
            simulator.run(idx)
            # save progress 
            simulationResult["RawSampleNum"] = idx+1
            with open(simulationResultPath, "w") as f:
                json.dump(simulationResult, f, indent=4)
        existedOutputNum = existedOutputNum + needAddOutputNum
        # calculate reflectance
        raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = postprocess.analyzeReflectance(sessionID, muaPath=muaPath)
        print("Session name: {} \nReflectance mean: {} \nCV: {} \nNecessary photon num: {:.4e}".format(sessionID, 
                                                                                                       reflectanceMean, 
                                                                                                       reflectanceCV, 
                                                                                                       totalPhoton*groupingNum), 
              end="\n\n")