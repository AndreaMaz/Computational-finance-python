#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test the Explicit Euler scheme and the Upwind method, computing the
prie of a call option. We investigate their stability.

@author: Andrea Mazzon
"""
import numpy as np
import math

from explicitEuler import ExplicitEuler
from upwind import Upwind

dx = 0.1
xmin = 0
xmax = 7

dt = 0.094*dx*dx 
#dt = 0.0947*dx*dx 
#dt = 0.095*dx*dx 
tmax = 3

sigma = lambda x : 0.5*x
r = 0.2

strike = 2

payoff = lambda x : np.maximum(x - strike, 0)

functionLeft = lambda x, t : 0
functionRight = lambda x, t : x - strike * math.exp(-r * t)

solver = ExplicitEuler(dx, dt, xmin, xmax, tmax, r, sigma, payoff, functionLeft, functionRight)
solver.solveAndPlot()

