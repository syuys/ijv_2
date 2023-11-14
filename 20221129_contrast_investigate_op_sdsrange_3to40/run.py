#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:17:21 2021

@author: md703
"""

from mcx_ultrasound_opsbased import MCX
from glob import glob
import postprocess
import json
import os
import shutil

# parameter
projectID = "20221129_contrast_investigate_op_sdsrange_3to40"
sessionIDSet = glob("ijv*")
gpuIdx = 1
cvThold = 0.07
repeatTimes = 10
muaTag = "mua*"
detectorNA = 0.59
projectOutputPath = f"/home/md703/syu/ijv_2_output/{projectID}"
host = "md703@192.168.31.104"  # original: md703@192.168.31.237
hostFolder = f"/media/md703/Expansion/syu/ijv_2_output/{projectID}"


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
        tmp= tmp["MovingAverageGroupingSampleCV"].values()
        tmp = list(tmp)
        cv.append(tmp)
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
        # existNum = 0
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
    cvRecords[sessionID] = cvSet.max()
    
    # # send new jdat and result.json (or backup new jdat)
    # status = os.system(f"sshpass -p md703 scp -r {outputPath} {host}:{hostFolder}")
    # if status != 0:  # if scp is failed
    #     backupPath = os.path.join(outputPath, "mcx_output", "backup")
    #     if not os.path.isdir(backupPath):
    #         os.mkdir(backupPath)
    #     detOutputPathSet.sort(key=lambda x: int(x.split("_")[-2]))
    #     for detOutputPath in detOutputPathSet[-repeatTimes:]:
    #         shutil.copy(detOutputPath, backupPath)
    
    # copy to backup folder (checking and sending are done by another code.)    
    backupPath = os.path.join(outputPath, "mcx_output", "backup")
    if not os.path.isdir(backupPath):
        os.mkdir(backupPath)
    for detOutputPath in detOutputPathSet[-repeatTimes:]:
        shutil.copy(detOutputPath, backupPath)


# # when simulation is done, remove the rest jdata.
# for sessionID in sessionIDSet:
#     with open(os.path.join(sessionID, "config.json")) as f:
#         config = json.load(f)
#     detOutputPathSet = glob(os.path.join(config["OutputPath"], sessionID, 
#                                          "mcx_output", "*.jdat"))
#     for detOutputPath in detOutputPathSet:
#         os.remove(detOutputPath)