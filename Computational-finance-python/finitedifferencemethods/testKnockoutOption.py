#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It tests the pricing of a Knock-out (down-and-out) option, both with 
Monte-Carlo simulation of the underlying process and PDEs, in the case of a
Black-Scholes model.

@author: Andrea Mazzon
"""

import math
import time
import numpy as np

from processSimulation.eulerDiscretizationForBlackScholes import EulerDiscretizationForBlackScholes
from optionMonteCarloValuation.knockOutOption import KnockOutOption
from analyticformulas.analyticFormulas import blackScholesDownAndOut
from implicitEuler import ImplicitEuler



maturity = 3

initialValue = 2
r = 0.0
sigma = 0.5 


strike = 2
lowerBarrier = 0.2

analyticPrice = blackScholesDownAndOut(initialValue, r, sigma, maturity, strike, lowerBarrier)

print("The analytic price is ", analyticPrice)

#Monte-Carlo

numberOfSimulations = 10000
seed = 1897

timeStep = 0.01
finalTime = maturity

payoffFunction = lambda x : np.maximum(x-strike,0)

transform = lambda x : math.exp(x)
inverseTransform = lambda x : math.log(x)

timeMCInit = time.time() 

eulerBlackScholes= EulerDiscretizationForBlackScholes(numberOfSimulations, timeStep, finalTime, 
                   initialValue, r, sigma)

processRealizations = eulerBlackScholes.getRealizations()

knockOutOption = KnockOutOption(payoffFunction, maturity, lowerBarrier)

MCprice = knockOutOption.getPrice(processRealizations)

timeNeededMC = time.time()  - timeMCInit

print("The Monte-Carlo price is ", MCprice)
print("The time needed with Monte-Carlo is ", timeNeededMC)

#Implicit Euler

dx = 0.05

xmin = lowerBarrier
xmax = 13

dt = dx 
tmax = 3

sigmaFunction = lambda x : sigma*x

functionLeft = lambda x, t : 0
functionRight = lambda x, t : x - strike * math.exp(-r * t)

timeImplicitInit = time.time() 

implicitEulerSolver = ImplicitEuler(dx, dt, xmin, xmax, tmax, r, sigmaFunction, payoffFunction, functionLeft, functionRight)

priceIE = implicitEulerSolver.getSolutionForGivenTimeAndValue(tmax, initialValue)

timeNeededImplicit = time.time()  - timeImplicitInit

print("The price with Implicit Euler is  ", priceIE)
print("The time needed with Implicit Euler is  ", timeNeededImplicit)
