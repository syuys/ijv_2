#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:05:19 2022

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import jdata as jd
import os
from glob import glob
import json
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


sessionIDSet = ["ijv_dis_mus_lb_bloodG_low", "ijv_col_mus_lb_bloodG_low"]

for sessionID in sessionIDSet:
    with open(os.path.join(sessionID, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(sessionID, "model_parameters.json")) as f:
        modelParameters = json.load(f)  # about index of materials & fiber number
    with open(config["MCXInputPath"]) as f:
        mcxInputTemplate = json.load(f)
    fiberSet = modelParameters["HardwareParam"]["Detector"]["Fiber"]
    detectorNA=config["DetectorNA"]
    detOutputPathSet = glob(os.path.join(config["OutputPath"], sessionID, "mcx_output", "*.jdat"))  # about paths of detected photon data
    innerIndex=modelParameters["OptParam"]["Prism"]["n"]
    outerIndex=modelParameters["OptParam"]["Prism"]["n"]
    detectorNum=len(modelParameters["HardwareParam"]["Detector"]["Fiber"])*3*2
    
    validPPath = np.empty((0, len(mcxInputTemplate["Domain"]["Media"])-1))
    for detOutputIdx, detOutputPath in enumerate(detOutputPathSet):
        # read detected data
        detOutput = jd.load(detOutputPath)
        info = detOutput["MCXData"]["Info"]
        photonData = detOutput["MCXData"]["PhotonData"]
        
        # unit conversion for photon pathlength
        photonData["ppath"] = photonData["ppath"] * info["LengthUnit"]
        
        # retrieve valid detector ID and valid ppath
        critAng = np.arcsin(detectorNA/innerIndex)
        afterRefractAng = np.arccos(abs(photonData["v"][:, 2]))
        beforeRefractAng = np.arcsin(outerIndex*np.sin(afterRefractAng)/innerIndex)
        validPhotonBool = beforeRefractAng <= critAng
        validDetID = photonData["detid"][validPhotonBool]
        validDetID = validDetID - 1  # make detid start from 0
        validPPath = np.concatenate((validPPath, photonData["ppath"][validPhotonBool]), axis=0)
        