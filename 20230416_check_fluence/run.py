#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:17:21 2021

@author: md703
"""

from mcx_ultrasound_opsbased import MCX
from glob import glob
import json
import os
import shutil

# parameter
projectID = "20230416_check_fluence"
sessionIDSet = [
    # "ijv_dis_depth_-2_std_src", "ijv_dis_depth_-2_std_det20",                   
    # "ijv_col_depth_-2_std_src", "ijv_col_depth_-2_std_det20",
    "ijv_dis_depth_+0_std_src", "ijv_dis_depth_+0_std_det20",
    "ijv_col_depth_+0_std_src", "ijv_col_depth_+0_std_det20",
    ]
gpuIdx = 2
repeatTimes = 10
projectOutputPath = f"/home/md703/syu/ijv_2_output/{projectID}"


# Initialize
if not os.path.isdir(projectOutputPath):
    os.mkdir(projectOutputPath)

simulators = {}  # simulator for each session
for sessionID in sessionIDSet:
    # simulator
    simulators[sessionID] = MCX(sessionID, wmc=False)


# run 
while True:    
    for sessionID in sessionIDSet:
        
        # calculate sim num
        with open(os.path.join(sessionID, "config.json")) as f:
            config = json.load(f)
        outputPath = os.path.join(config["OutputPath"], sessionID)
        existNum = glob(os.path.join(outputPath, "mcx_output", "*.jnii"))
        existNum = [int(i.split("_")[-1].split(".")[0]) for i in existNum]
        if len(existNum) == 0:
            existNum = 0
        else:
            existNum = max(existNum)+1
        needAddNum = repeatTimes - existNum % repeatTimes
        
        # run forward mcx
        for idx in range(existNum, existNum+needAddNum):
            simulators[sessionID].run(idx, gpuIdx=gpuIdx)
        
        # if jnii num > 10, delete old 10 jnii
        detOutputPathSet = glob(os.path.join(outputPath, "mcx_output", "*.jnii"))
        detOutputPathSet.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(detOutputPathSet) > repeatTimes:        
            for detOutputPath in detOutputPathSet[:repeatTimes]:
                os.remove(detOutputPath)
    
        
        # copy to backup folder (checking and sending are done by another code.)    
        backupPath = os.path.join(outputPath, "mcx_output", "backup")
        if not os.path.isdir(backupPath):
            os.mkdir(backupPath)
        for detOutputPath in detOutputPathSet[-repeatTimes:]:
            shutil.copy(detOutputPath, backupPath)