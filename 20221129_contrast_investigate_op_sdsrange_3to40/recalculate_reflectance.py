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
sessionIDSet = ["ijv_dis_mus_20%", "ijv_col_mus_20%", "ijv_dis_mus_40%", "ijv_col_mus_40%"]
muaTag = "mua*"
detectorNA = 0.59
hostFolder = "/media/md703/Expansion/syu/ijv_2_output/20221129_contrast_investigate_op_sdsrange_3to40"

# calculate
for sessionID in sessionIDSet:
    print(f"sessionID: {sessionID}")
    
    # load mua for calculating reflectance
    muaPathSet = glob(os.path.join(sessionID, muaTag))
    
    raw, maR, maRM, maRCV, photon, groupingNum = postprocess.analyzeReflectance(sessionID, muaPathSet, detectorNA)