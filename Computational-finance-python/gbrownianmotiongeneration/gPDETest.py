#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

from gPDESolution import GPDESolution
from gPDESolutionImplicit import GPDESolutionImplicit

dx = 0.01

xmin = -7.0
xmax = 7.0

sigmaDown = 1
sigmaUp = 1

dt = dx*dx/sigmaUp**2
tmax = 1

threshold = 0

pdeSolver = GPDESolution(dx, dt, xmin, xmax, tmax, sigmaDown, sigmaUp)

pdeSolver.setThresholdForInitialCondition(0.0)

print("Solution with explicit Euler: ", pdeSolver.getSolutionForGivenTimeAndValue(1, 0.0))


pdeSolverImpl = GPDESolutionImplicit(dx, dx, xmin, xmax, tmax, sigmaDown, sigmaUp)

pdeSolverImpl.setThresholdForInitialCondition(0.0)

print("Solution with implicit Euler: ", pdeSolverImpl.getSolutionForGivenTimeAndValue(1, 0.0))