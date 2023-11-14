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
                # "ijv_dis_depth_-1_std",
                # "ijv_col_depth_-1_std",
                # "ijv_dis_depth_+0_std",
                # "ijv_col_depth_+0_std",
                # "ijv_dis_depth_+1_std",
                "ijv_col_depth_+1_std"
                ]
muaTag = "mua*"
detectorNA = 0.22

# calculate
for sessionID in sessionIDSet:
    print(f"sessionID: {sessionID}")
    
    # load mua for calculating reflectance
    muaPathSet = glob(os.path.join(sessionID, muaTag))
    
    raw, maR, maRM, maRCV, photon, groupingNum = postprocess.analyzeReflectance(sessionID, muaPathSet, detectorNA)