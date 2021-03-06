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

from eulerDiscretizationForBlackScholes import EulerDiscretizationForBlackScholes
from optionMonteCarloValuation.knockOutOption import KnockOutOption
from analyticformulas.analyticFormulas import blackScholesDownAndOut
from finitedifferencemethods.implicitEuler import ImplicitEuler



numberOfSimulations = 10000
seed = 1897

timeStep = 0.001
finalTime = 3
maturity = finalTime

initialValue = 2
r = 0.0
sigma = 0.5 


strike = 2
lowerBarrier = 1.2
#upperBarrier = 3.2

analyticPrice = blackScholesDownAndOut(initialValue, r, sigma, maturity, strike, lowerBarrier)

print("The analytic price is ", analyticPrice)

#Monte-Carlo

payoffFunction = lambda x : np.maximum(x-strike,0)

transform = lambda x : math.exp(x)
inverseTransform = lambda x : math.log(x)

timeMCInit = time.time() 

eulerBlackScholes= EulerDiscretizationForBlackScholes(numberOfSimulations, timeStep, finalTime, 
                   initialValue, r, sigma)

processRealizations = eulerBlackScholes.getRealizations()

knockOutOption = KnockOutOption(payoffFunction, maturity, lowerBarrier)

timeNeededMC = time.time()  - timeMCInit

print("The Monte-Carlo price is ", knockOutOption.getPrice(processRealizations))
print("The time needed with Monte-Carlo is ", timeNeededMC)

#Implicit Euler

dx = 0.01

xmin = lowerBarrier
xmax = 13
#xmax = upperBarrier

dt = dx 
tmax = 3

sigmaFunction = lambda x : sigma*x

functionLeft = lambda x, t : 0
#functionRight = lambda x, t : 0
functionRight = lambda x, t : x - strike * math.exp(-r * t)

implicitEulerSolver = ImplicitEuler(dx, dt, xmin, xmax, tmax, r, sigmaFunction, payoffFunction, functionLeft, functionRight)

timeImplicitInit = time.time() 
price = implicitEulerSolver.getSolutionForGivenMaturityAndValue(tmax, initialValue)
timeNeededImplicit = time.time()  - timeImplicitInit

print("The price with Implicit Euler is  ", price)
print("The time needed with Implicit Euler is  ", timeNeededImplicit)
