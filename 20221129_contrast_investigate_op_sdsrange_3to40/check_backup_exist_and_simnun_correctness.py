#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:04:14 2023

@author: md703
"""

import numpy as np
import postprocess
from glob import glob
import os
import json


# parameters
root = "/media/md703/Expansion/syu/ijv_2_output"
simroot = "/home/md703/syu/ijv_2"
projectID = "20221129_contrast_investigate_op_sdsrange_3to40"
sessionPathSet = glob(os.path.join(root, projectID, "ijv*"))
detectorNA = 0.59


# %% find which session have backup
for sessionPath in sessionPathSet:
    backupPath = os.path.join(sessionPath, "mcx_output", "backup")
    sessionID = os.path.split(sessionPath)[-1]
    if os.path.isdir(backupPath):
        print(f"{sessionID}: Find !")
    else:
        print(f"{sessionID}:  -")

print("\n")


# %% total jdata num match order ?
for sessionPath in sessionPathSet:
    sessionID = os.path.split(sessionPath)[-1]
    jdatSet = glob(os.path.join(sessionPath, "mcx_output", "*.jdat"))
    orderSet = [int(jdat.split("_")[-2]) for jdat in jdatSet]
    
    if (max(orderSet) == len(jdatSet)-1) & (min(orderSet) == 0):
        print(f"{sessionID}: jdata num match order.")
        
    else:
        print(f"{sessionID}: NOT MATCH !")

print("\n")


# %% every jdat size is similar ?
for sessionPath in sessionPathSet:
    sessionID = os.path.split(sessionPath)[-1]
    jdatSet = glob(os.path.join(sessionPath, "mcx_output", "*.jdat"))
    sizeSet = [os.path.getsize(jdat)/1e6 for jdat in jdatSet]  # MB
    
    m = min(sizeSet)
    M = max(sizeSet)
    print(f"{sessionID} - min: {m}, max: {M}, diff: {np.around((M-m)/M*100, 2)}%")

print("\n")


# %% total jdata num = analyzed num ?
for sessionPath in sessionPathSet:
    sessionID = os.path.split(sessionPath)[-1]
    jdatSet = glob(os.path.join(sessionPath, "mcx_output", "*.jdat"))
    resultPath = glob(os.path.join(sessionPath, "post_analysis", "*.json"))[0]
    with open(resultPath) as f:
        result = json.load(f)
    recordSimNum = result["AnalyzedSampleNum"]
    if len(jdatSet) == recordSimNum:
        print(f"{sessionID}: equal !")
    else:
        print(f"{sessionID}: NOT EQUAL !")
        print(f"jdatnum: {len(jdatSet)},  analyzednum: {recordSimNum}\n")
        
        # # re-calculate reflectance and update result.json
        # muaPathSet = glob(os.path.join(simroot, projectID, sessionID, "mua*"))
        # sim = postprocess.analyzeReflectance(sessionID, muaPathSet, detectorNA, 
        #                                       updateResultFile=True, 
        #                                       showCvVariation=False)

print("\n")