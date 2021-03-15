#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test the simulation of the G-Brownian motion

@author: Andrea Mazzon
"""

from gBrownianMotion import GBrownianMotion

#the discretization for the solution of the PDE
dx = 0.01

xmin = - 5
xmax = 5

dt = dx*dx 
tmax = 1

#the discretization where we compute the first distributions
minusA = - 5
plusA = 5
dxFirstDistributions = 0.05

#the uncertainty interval
sigmaDown = 1
sigmaUp = 1

#parameters for the simulation
dtBrownianIncrements = 0.01
numberOfSimulations = 1000

gBrownianMotion = GBrownianMotion(dx, dt, xmin, xmax, tmax, minusA, plusA, dtBrownianIncrements,
                                  dxFirstDistributions, sigmaDown, sigmaUp,
                                  numberOfSimulations)

#we want to store the realizations, so that we can have a look at them
realizations = gBrownianMotion.getGBrownianMotion()


gBrownianMotion.plotPaths(0,10)

print("The average of the realizations is ", gBrownianMotion.getAverageAtGivenTime(tmax))