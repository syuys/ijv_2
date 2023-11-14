#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 02:41:43 2022

@author: md703
"""

import numpy as np
import os
import json
import postprocess
from postprocessor import postprocessor
import time

sessionID = "large_ijv_mus_baseline"
with open(os.path.join(sessionID, "mua.json")) as f:
    mua = json.load(f)
muaUsed =[mua["1: Air"],
          mua["2: PLA"],
          mua["3: Prism"],
          mua["4: Skin"],
          mua["5: Fat"],
          mua["6: Muscle"],
          mua["7: Muscle or IJV (Perturbed Region)"],
          mua["8: IJV"],
          mua["9: CCA"]
          ]

## Test
repeatTimes = 5
# old way
start_time = time.time()
for _ in range(repeatTimes):
    meanPathlength_old, movingAverageMeanPathlength_old = postprocess.getMeanPathlength(sessionID, mua=muaUsed)
print("Old: %s seconds" % (time.time() - start_time))
# print("Pathlength mean:", movingAverageMeanPathlength_old)
print()

# new way
start_time = time.time()
calculator = postprocessor(sessionID)
for _ in range(repeatTimes):
    meanPathlength_new, movingAverageMeanPathlength_new = calculator.getMeanPathlength(muaUsed)
print("New: %s seconds" % (time.time() - start_time))
# print("Pathlength mean:", movingAverageMeanPathlength_new)

print()
print("Two kinds of pathlength are the same: ", np.all(movingAverageMeanPathlength_old == movingAverageMeanPathlength_new))