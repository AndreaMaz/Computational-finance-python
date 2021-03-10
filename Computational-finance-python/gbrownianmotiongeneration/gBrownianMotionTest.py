#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test the simulation of the G-Brownian motion

@author: Andrea Mazzon
"""

from gBrownianMotion import GBrownianMotion

dx = 0.005
xmin = -0.5
xmax = 0.5

dt = 0.5*dx*dx 
tmax = 1

sigmaDown = 0.5
sigmaUp = 1

dtBrownianIncrements = 0.005
dxFirstDistributions = 0.1

threshold = 0

numberOfSimulations = 6

gBrownianMotion = GBrownianMotion(dx, dt, xmin, xmax, tmax, dtBrownianIncrements,
                                  dxFirstDistributions, sigmaDown, sigmaUp,
                                  numberOfSimulations)


realizations = gBrownianMotion.getGBrownianMotion()


gBrownianMotion.plotPaths(0,5)

