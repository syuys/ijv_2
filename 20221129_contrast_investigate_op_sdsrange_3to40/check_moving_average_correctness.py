#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:01:43 2023

@author: md703
"""

import postprocess_old as old
import postprocess as new
from glob import glob
import os

sessionID = "ijv_col_mus_0%"
muaPathSet = glob(os.path.join(sessionID, "mua*"))
detectorNA = 0.59

rawRef_o, movFnlRef_o, movFnlRefMean_o, movFnlRefCV_o, pN_o, fnlGrpNum_o = old.analyzeReflectance(sessionID, 
                                                                                                  muaPathSet, 
                                                                                                  detectorNA, 
                                                                                                  updateResultFile=False, 
                                                                                                  showCvVariation=False)
print("old finished")

rawRef_n, movFnlRef_n, movFnlRefMean_n, movFnlRefCV_n, pN_n, fnlGrpNum_n = new.analyzeReflectance(sessionID, 
                                                                                                  muaPathSet, 
                                                                                                  detectorNA, 
                                                                                                  updateResultFile=False, 
                                                                                                  showCvVariation=False)
print("new finished")

print(f"movFnlRefMean_o[0, 0]: {movFnlRef_o[0, 0]}")
print(f"movFnlRefMean_n[0, 0]: {movFnlRef_n[0, 0]}")