#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 02:41:43 2022

@author: md703
"""

import os
import json
import postprocess
from postprocessor import postprocessor
import time

sessionID = "I_730"
with open(os.path.join(sessionID, "mua.json")) as f:
    mua = json.load(f)
muaUsed =[mua["1: Air"],
          mua["2: PLA"],
          mua["3: Prism"],
          mua["4: Phantom body"]
          ]

## Test
repeatTimes = 5
# old way
start_time = time.time()
for _ in range(repeatTimes):
    raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = postprocess.analyzeReflectance(sessionID, mua=muaUsed)
print("Old: %s seconds" % (time.time() - start_time))
print("Reflectance mean:", reflectanceMean)
print()

# new way
start_time = time.time()
calculator = postprocessor(sessionID)
for _ in range(repeatTimes):
    raw, reflectance, reflectanceMean, reflectanceCV, totalPhoton, groupingNum = calculator.analyzeReflectance(muaUsed)
print("New: %s seconds" % (time.time() - start_time))
print("Reflectance mean:", reflectanceMean)