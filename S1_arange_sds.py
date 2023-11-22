#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:56:35 2022

@author: md703
"""
import numpy as np
import json
import os
from glob import glob


# %% parameters
projectID = "20230911_check_led_pattern_sdsrange_5to45_g99"
sessionIDSet = [i.split("/")[-1] for i in glob(os.path.join(projectID, "ijv*"))]  # *skin*fat*, "ijv*"
sdsfilepath = "model_parameters.json"

fiber = []
radius = 0.3675  # mm
sdsInterval = radius*2
startpos = 5  # mm,  3
limit = 45  # mm,  40


# %% arange sds
while startpos < limit:
    startpos = float(np.around(startpos, 3))
    print(f"pos: {startpos}")
    
    fiber.append({"SDS": startpos, 
                  "Radius": radius
                  })
    
    startpos += sdsInterval

if len(sessionIDSet) == 0:
    raise Exception("Path error in sessionID !")
for sessionID in sessionIDSet:
    path = os.path.join(projectID, sessionID, sdsfilepath)
    with open(path) as f:
        sdsfile = json.load(f)
    sdsfile["HardwareParam"]["Detector"]["Fiber"] = fiber
    with open(path, "w") as f:
        json.dump(sdsfile, f, indent=4)
    print(f"sessionID: {sessionID}")
    
    
    