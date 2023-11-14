#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:27:24 2023

@author: md703
"""

from glob import glob
import os
import json

sessionIDSet = glob("*%")

for sessionID in sessionIDSet:
    muaPath = os.path.join(sessionID, "mua_40%.json")
    with open(muaPath) as f:
        mua = json.load(f)
        
    