#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

from gPDESolution import GPDESolution

dx = 0.005
xmin = -2
xmax = 2

dt = 0.5*dx*dx 
tmax = 1

sigmaDown = 0.6
sigmaUp = 1

threshold = 0

pdeSolver = GPDESolution(dx, dt, xmin, xmax, tmax, sigmaDown, sigmaUp)

pdeSolver.setThresholdForInitialCondition(threshold)

print(pdeSolver.getSolutionForGivenMaturityAndValue(tmax, 0))