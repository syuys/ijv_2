#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:46:16 2022

@author: md703
"""

import numpy as np
import jdata as jd

detOutput = jd.load("/home/md703/Desktop/ijv_2_output/20221019_contrast_test_enlarge_sdsrange/ijv_col_mus_lb_bloodG_low/mcx_output/ijv_col_mus_lb_bloodG_low_0_detp.jdat")
photonData = detOutput["MCXData"]["PhotonData"]

outerIndex = 1.51
innerIndex = 1.51

afterAng = np.arccos(abs(photonData["v"][:, 2]))
beforeAng = np.arcsin(outerIndex*np.sin(afterAng)/innerIndex)
beforeAng = np.rad2deg(beforeAng)

print(f"Max ang: {max(beforeAng)}")
print(f"Min ang: {min(beforeAng)}")








