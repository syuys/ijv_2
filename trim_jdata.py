#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:21:15 2023

@author: md703
"""

import jdata as jd
import postprocess
from copy import deepcopy
import os
from glob import glob

# detOutputPath = "/home/md703/Desktop/c.jdat"
# detOutput = jd.load(detOutputPath)
# detOutputNew = postprocess.trimDivergentPhoton(deepcopy(detOutput), 
#                                                 innerIndex=1.51, 
#                                                 outerIndex=1.457, 
#                                                 detectorNA = 0.59)
# postprocess.trimJdata(detOutputPath, 
#                       innerIndex=1.51, 
#                       outerIndex=1.457, 
#                       detectorNA = 0.59)

# projectIDSet = glob(os.path.join("/media/md703/Expansion/syu/ijv_2_output", "*"))
projectIDSet = ["/media/md703/Expansion/syu/ijv_2_output/20221218_contrast_investigate_ijvdepth_sdsrange_3to40"]
for projectID in projectIDSet:
    simTypeSet = glob(os.path.join(projectID, "*"))
    simTypeSet.remove('/media/md703/Expansion/syu/ijv_2_output/20221218_contrast_investigate_ijvdepth_sdsrange_3to40/ijv_col_depth_-2_std')
    simTypeSet.remove('/media/md703/Expansion/syu/ijv_2_output/20221218_contrast_investigate_ijvdepth_sdsrange_3to40/ijv_dis_depth_-1_std')
    simTypeSet.remove('/media/md703/Expansion/syu/ijv_2_output/20221218_contrast_investigate_ijvdepth_sdsrange_3to40/ijv_col_depth_+1_std')
    for simType in simTypeSet:
        print(simType)
        detOutputPathSet = glob(os.path.join(simType, "mcx_output", "*.jdat"))
        for detOutputPath in detOutputPathSet:
            detOutput = postprocess.trimJdata(detOutputPath, 
                                              innerIndex=1.51, 
                                              outerIndex=1.457, 
                                              detectorNA = 0.59)