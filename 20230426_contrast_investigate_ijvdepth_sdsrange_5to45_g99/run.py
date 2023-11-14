#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:17:21 2021

@author: md703
"""

from mcx_ultrasound_opsbased import MCX
import numpy as np
from glob import glob
import postprocess
import json
import os
import shutil

# parameter
projectID = "20230426_contrast_investigate_ijvdepth_sdsrange_5to45_g99"
sessionIDSet = [
                "ijv_dis_depth_-1.6_std", "ijv_col_depth_-1.6_std",
                # "ijv_dis_depth_+0_std", "ijv_col_depth_+0_std",
                # "ijv_dis_depth_+1_std", "ijv_col_depth_+1_std",
                # "ijv_dis_depth_+2_std", "ijv_col_depth_+2_std",
                ]
gpuIdx = 2
sdsCVobserveEnd = 30.725  # enter the right length in mm
cvThold = 0.001
repeatTimes = 10
muaTag = "mua*"
detectorNA = 0.22
projectOutputPath = f"/home/md703/syu/ijv_2_output/{projectID}"


# Initialize
if not os.path.isdir(projectOutputPath):
    os.mkdir(projectOutputPath)

simulators = {}  # simulator for each session
cvRecords = {}  # store max cv of each session
for sessionID in sessionIDSet:
    # simulator
    simulators[sessionID] = MCX(sessionID)
    # cv record
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)
    cv = []
    muaPathSet = glob(os.path.join(sessionID, muaTag))
    for muaPath in muaPathSet:
        muaType = muaPath.split("/")[-1][:-5]
        with open(os.path.join(config["OutputPath"], sessionID, "post_analysis", f"{sessionID}_simulation_result_{muaType}.json")) as f:
            tmp = json.load(f)
        sdsCVobserveEndIdx = list(tmp["MovingAverageGroupingSampleCV"].keys())
        sdsCVobserveEndIdx = np.where(np.array(sdsCVobserveEndIdx) == f"sds_{sdsCVobserveEnd}")[0][0]
        tmp= list(tmp["MovingAverageGroupingSampleCV"].values())
        cv.append(tmp[:sdsCVobserveEndIdx+1])
    cvRecords[sessionID] = max(map(max, cv))


# run the session which has higher cv
while max(cvRecords.values()) > cvThold:
    # show recent cv progress
    print()
    for sessionID in sessionIDSet:
        print(f"{sessionID}, cv: {cvRecords[sessionID]}")
    print()
    
    # select target
    sessionID = max(cvRecords, key=cvRecords.get)
    
    # load mua for calculating reflectance
    muaPathSet = glob(os.path.join(sessionID, muaTag))
    
    # calculate sim num
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)
    outputPath = os.path.join(config["OutputPath"], sessionID)
    existNum = glob(os.path.join(outputPath, "mcx_output", "*.jdat"))
    existNum = [int(i.split("_")[-2]) for i in existNum]
    if len(existNum) == 0:
        muaType = glob(os.path.join(sessionID, muaTag))[0].split("/")[-1][:-5]
        with open(os.path.join(outputPath, "post_analysis", f"{sessionID}_simulation_result_{muaType}.json")) as f:
            result = json.load(f)
        existNum = result["AnalyzedSampleNum"]
    else:
        existNum = max(existNum)+1
    needAddNum = repeatTimes - existNum % repeatTimes
    
    # run forward mcx
    for idx in range(existNum, existNum+needAddNum):
        simulators[sessionID].run(idx, gpuIdx=gpuIdx)
    
    # if jdat num > 10, delete old jdat to host server
    detOutputPathSet = glob(os.path.join(outputPath, "mcx_output", "*.jdat"))
    detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
    if len(detOutputPathSet) > repeatTimes:        
        for detOutputPath in detOutputPathSet[:repeatTimes]:
            os.remove(detOutputPath)
    
    # update reflectance and cv in result.json
    cvSet = postprocess.updateReflectance(sessionID, muaPathSet=muaPathSet, 
                                          detectorNA=detectorNA)
    cvRecords[sessionID] = cvSet[:, :sdsCVobserveEndIdx+1].max()
    
    # copy to backup folder (checking and sending are done by another code.)    
    backupPath = os.path.join(outputPath, "mcx_output", "backup")
    if not os.path.isdir(backupPath):
        os.mkdir(backupPath)
    for detOutputPath in detOutputPathSet[-repeatTimes:]:
        shutil.copy(detOutputPath, backupPath)