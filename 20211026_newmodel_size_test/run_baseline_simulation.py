#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 14:40:14 2021

@author: md703
"""

from mcx_ultrasound_model import MCX

# parameters
sessionID = "test_bc_eric_smallijv_mus_lb"

# initialize
simulator = MCX(sessionID)

# run forward mcx
simulator.run()