#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 21:59:26 2022

@author: md703
"""

import os
import json
from glob import glob
import numpy as np

# %% parameters
projectID = "20230911_check_led_pattern_sdsrange_5to45_g99"
pathIDSet = glob(os.path.join(projectID, "ijv*"))  # *skin*fat*, ijv*
subject = "EU"
neckPos = "UpperNeck"
geoBoundPath = "shared_files/model_input_related/geo_bound.json"
modelParamPath = "model_parameters.json"
# stdtimesWay = 0  # eval(" float(pathID.split('_')[-2]) ")
modelX = 240  # mm
modelY = 120  # mm
modelZ =  80  # mm

# %% load
with open(geoBoundPath) as f:
    geoBound = json.load(f)
ijvDepthAve = geoBound["IJV"]["Depth"]["Average"]
ijvDepthStd = geoBound["IJV"]["Depth"]["Std"]
ijvMajorAve = geoBound["IJV"]["MajorAxisNormal"]["Average"]
ijvMajorStd = geoBound["IJV"]["MajorAxisNormal"]["Std"]
ijvMinorAve = geoBound["IJV"]["MinorAxisNormal"]["Average"]
ijvMinorStd = geoBound["IJV"]["MinorAxisNormal"]["Std"]

# iteratively adjust ijv depth parameter
if len(pathIDSet) == 0:
    raise Exception("Error in pathIDSet !")

for pathID in pathIDSet:
    sessionID = pathID.split('/')[-1]
    print(f"\nsessionID: {sessionID}")
    
    if sessionID.split("_")[2] == subject:
        with open(os.path.join("ultrasound_image_processing/subject_neck_image_202305~", subject, f"{subject}_geo.json")) as f:
            geoSubject = json.load(f)[neckPos]
        # ijv
        ijvDepth = geoSubject["IJV"]["Depth"]
        ijvMajor = geoSubject["IJV"]["MajorAxisNormal"]
        ijvMinor = geoSubject["IJV"]["MinorAxisNormal"]
        majorAxisChangePct = geoSubject["IJV"]["MajorAxisChangePct"]
        minorAxisChangePct = geoSubject["IJV"]["MinorAxisChangePct"]
        
        # cca
        cca_radius = geoSubject["CCA"]["Radius"]
        cca_sDist = geoSubject["CCA"]["sDist"]
        cca_sAng = geoSubject["CCA"]["sAng"]
        
    else:
        ## para
        # ijv depth
        if sessionID.split("_")[2] == "depth":
            ijvDepthTimes = float(sessionID.split("_")[3])
        else:
            ijvDepthTimes = 0    
        ijvDepth = np.around(ijvDepthAve + ijvDepthStd * ijvDepthTimes, 6)
        print(f"IJV std: {ijvDepthTimes}, depth: {ijvDepth}")
        
        # ijv major axis and minor axis
        if pathID.split("_")[-3] == "major":
            ijvMajorTimes = float(pathID.split("_")[-2])  # for std
            ijvMajor = np.around(ijvMajorAve + ijvMajorStd * ijvMajorTimes, 6)
            ijvMinor = ijvMinorAve
        elif pathID.split("_")[-3] == "minor":
            ijvMinorTimes = float(pathID.split("_")[-2])  # for mm
            ijvMinor = np.around(ijvMinorAve + 1 * ijvMinorTimes, 6)
            ijvMajor = ijvMajorAve
        else:
            ijvMajor = ijvMajorAve
            ijvMinor = ijvMinorAve
        print(f"IJV minor: {ijvMinor}, major: {ijvMajor}")
        
        # other ijv parameters
        majorAxisChangePct = geoBound["IJV"]["MajorAxisChangePct"]["Average"]
        minorAxisChangePct = geoBound["IJV"]["MinorAxisChangePct"]["Average"]
        
        # cca sAng
        if sessionID.split("_")[2] == "ccasAng":
            cca_sAng = float(sessionID.split("_")[3])
        else:
            cca_sAng = geoBound["CCA"]["sAng"]["Average"]
        print(f"cca_sAng: {cca_sAng}")
        
        # other cca paramters
        cca_radius = geoBound["CCA"]["Radius"]["Average"]
        cca_sDist = geoBound["CCA"]["sDist"]["Average"]
    
    
    # set geo        
    geo = {
        "Skin": {
            "__comment__": "in mm",
            "Thickness": geoBound["Skin"]["Thickness"]["Average"]
            },
        "Fat": {
            "__comment__": "in mm",
            "Thickness": geoBound["Fat"]["Thickness"]["Average"]
            },
        "IJV": {
            "__comment__": "ChangePct is from normal(mean) to large or small",
            "Depth": ijvDepth,
            "MajorAxisNormal":    ijvMajor,  # geoBound["IJV"]["MajorAxisNormal"]["Average"],
            "MinorAxisNormal":    ijvMinor,  # geoBound["IJV"]["MinorAxisNormal"]["Average"],
            "MajorAxisChangePct": majorAxisChangePct,
            "MinorAxisChangePct": minorAxisChangePct
            },
        "CCA": {
            "__comment__": "sAng is in degrees",
            "Radius": cca_radius,
            "sDist":  cca_sDist,
            "sAng":   cca_sAng
            }
        }
    
    with open(os.path.join(pathID, modelParamPath)) as f:
        modelParam = json.load(f)
    
    modelParam["GeoParam"] = geo
    
    # model size
    modelParam["ModelSize"]["XSize"] = modelX
    modelParam["ModelSize"]["YSize"] = modelY
    modelParam["ModelSize"]["ZSize"] = modelZ
    
    with open(os.path.join(pathID, modelParamPath), "w") as f:
        json.dump(modelParam, f, indent=4)
