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
sessionIDSet = [
                # "ijv_dis_HY_skin_50%_fat_50%_muscle_50%_blood_50%",  
                "ijv_col_HY_skin_50%_fat_50%_muscle_50%_blood_50%",
                ]
muaTag = "mua*"
detectorNA = 0.22

# calculate
for sessionID in sessionIDSet:
    print(f"sessionID: {sessionID}")
    
    # load mua for calculating reflectance
    muaPathSet = glob(os.path.join(sessionID, muaTag))
    
    raw, maR, maRM, maRCV, photon, groupingNum = postprocess.analyzeReflectance(sessionID, muaPathSet, detectorNA)