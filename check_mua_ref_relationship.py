#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:39:17 2023

@author: md703
"""

import json
import os
import numpy as np

path = "/media/md703/Expansion/syu/ijv_2_output/20221129_contrast_investigate_op_sdsrange_3to40/ijv_col_mus_0%/post_analysis"
files = ["ijv_col_mus_0%_simulation_result_mua_0%.json", "ijv_col_mus_0%_simulation_result_mua_20%.json", "ijv_col_mus_0%_simulation_result_mua_40%.json"]
reflectance = []

for file in files:
    with open(os.path.join(path, file)) as f:
        tmp = json.load(f)
    reflectance.append(list(tmp["MovingAverageGroupingSampleMean"].values()))

reflectance = np.array(reflectance)