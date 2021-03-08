#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It tests the pricing of a Knock-out (down-and-out) option, by using the 
Monte-Carlo simulation of the underlying process

@author: Andrea Mazzon
"""

import math
import time
import numpy as np

from processSimulation.eulerDiscretizationForBlackScholes import EulerDiscretizationForBlackScholes
from optionMonteCarloValuation.knockOutOption import KnockOutOption
from analyticformulas.analyticFormulas import blackScholesDownAndOut



numberOfSimulations = 10000
seed = 1897

timeStep = 0.1
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

