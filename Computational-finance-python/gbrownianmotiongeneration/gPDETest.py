#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

from gPDESolution import GPDESolution

dx = 0.01

xmin = -4.0
xmax = 4.0

sigmaDown = 1
sigmaUp = 1

dt = dx*dx/sigmaUp**2
tmax = 1

threshold = 0

pdeSolver = GPDESolution(dx, dt, xmin, xmax, tmax, sigmaDown, sigmaUp)

pdeSolver.setThresholdForInitialCondition(0.5)

print(pdeSolver.getSolutionForGivenTimeAndValue(1, 0.0))