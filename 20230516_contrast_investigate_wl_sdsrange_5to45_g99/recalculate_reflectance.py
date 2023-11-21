#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:52:11 2022

@author: md703
"""

import os
import postprocess
from glob import glob

# parameter
sessionIDSet = glob("ijv*")
muaTag = "mua_ijv*"
detectorNA = 0.22

# calculate
for sessionID in sessionIDSet:
    print(f"sessionID: {sessionID}")
    
    # load mua for calculating reflectance
    muaPathSet = glob(os.path.join(sessionID, muaTag))
    print(f"muaPathSet: {muaPathSet}")
    
    raw, maR, maRM, maRCV, photon, groupingNum = postprocess.analyzeReflectance(sessionID, muaPathSet, detectorNA)