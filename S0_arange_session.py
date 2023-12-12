#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:22:04 2022

@author: md703
"""

import os
import shutil


# %% parameters
preProjectID = "20230505_contrast_investigate_op_sdsrange_5to45_g99"
preSessionID = "ijv_dis_blood_0%"
curProjectID = "20231212_contrast_invivo_geo_simulation_cca_pulse"
curSessionIDSet = [
                    "ijv_dis_EU_skin_50%_fat_50%_muscle_50%_blood_50%",
                    "ijv_col_EU_skin_50%_fat_50%_muscle_50%_blood_50%",
                   ]

# %% make project ID folder and copy file (run.py ...etc)
if not os.path.isdir(curProjectID):
    os.mkdir(curProjectID)
if not os.path.isfile(os.path.join(curProjectID, "run.py")):
    shutil.copy(os.path.join(preProjectID, "run.py"), curProjectID)
if not os.path.isfile(os.path.join(curProjectID, "mua_template.json")):
    shutil.copy(os.path.join(preProjectID, "mua_template.json"), curProjectID)

# make sessionID folder and copy file (config, modelParameters)
prePath = os.path.join(preProjectID, preSessionID)
for curSessionID in curSessionIDSet:
    curPath = os.path.join(curProjectID, curSessionID)
    if not os.path.isdir(curPath):
        os.mkdir(curPath)
    shutil.copy(os.path.join(prePath, "config.json"), curPath)
    shutil.copy(os.path.join(prePath, "model_parameters.json"), curPath)