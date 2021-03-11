#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

from gPDESolution import GPDESolution
from gPDESolutionImplicit import GPDESolutionImplicit

dx = 0.01
xmin = -4.0
xmax = 4.0

dt = dx*dx 
tmax = 1

sigmaDown = 1*0.31622776601683794
sigmaUp = 1*0.31622776601683794

threshold = 0

pdeSolver = GPDESolution(dx, dt, xmin, xmax, tmax, sigmaDown, sigmaUp)

pdeSolver.setThresholdForInitialCondition(threshold)

print(pdeSolver.getSolutionForGivenTimeAndValue(1, 0.2))