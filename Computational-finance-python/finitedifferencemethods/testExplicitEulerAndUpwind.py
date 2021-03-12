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

sigma = 0.5
sigmaFunction = lambda x : sigma * x
r = 0.8

dt = 1.22*dx*dx /(sigma*xmax)**2

tmax = 2



strike = 2

payoff = lambda x : np.maximum(x - strike, 0)

functionLeft = lambda x, t : 0
functionRight = lambda x, t : x - strike * math.exp(-r * t)

solver = Upwind(dx, dt, xmin, xmax, tmax, r, sigmaFunction, payoff, functionLeft, functionRight)
solver.solveAndPlot()

