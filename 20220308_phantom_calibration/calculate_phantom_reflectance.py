#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 22:34:36 2022

@author: md703
"""

from glob import glob
import os
import numpy as np
from postprocessor import postprocessor
import json
from tqdm import tqdm


#sessionIDSet = [f"{phantom}_{wl}" for phantom in "C" for wl in ["730", "760", "780", "810", "850"]]
#for sessionID in tqdm(sessionIDSet):
sessionID = "C_850"
with open(os.path.join(sessionID, "mua.json")) as f:
    mua = json.load(f)
muaUsed =[mua["1: Air"],
          mua["2: PLA"],
          mua["3: Prism"],
          mua["4: Phantom body"]
          ]

calculator = postprocessor(sessionID)
raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = calculator.analyzeReflectance(muaUsed)
np.save(f"/home/md703/Desktop/ijv_2_output/20220308_phantom_calibration/reflectance/{sessionID}",  reflectanceMean)
    