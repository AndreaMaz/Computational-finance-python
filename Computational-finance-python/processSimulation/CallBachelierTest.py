#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class we do some first tests for the use of Monte-Carlo methods to 
simulate continuous time stochastic processes.

@author: Andrea Mazzon
"""
import numpy as np
import math

from eulerDiscretizationForBlackScholes import EulerDiscretizationForBlackScholes
from standardEulerDiscretization import StandardEulerDiscretization
from optionMonteCarloValuation.simpleEuropeanOption import SimpleEuropeanOption
from analyticformulas.analyticFormulas import blackScholesPriceCall


numberOfSimulations = 10000
timeStep = 0.1
finalTime = 3

initialValue = math.log(3)
r = 0.0
sigma = 0.8


maturity = finalTime
strike = initialValue


muFunction = lambda t, x : 0
sigmaFunction = lambda t, x : sigma

payoffFunction = lambda x : max(x-strike,0)


transformErrors = []
standardErrors = []


#price and error generating the process by simulating the logarithm 



standardEuler = StandardEulerDiscretization(numberOfSimulations, timeStep, finalTime, 
                   initialValue, muFunction, sigmaFunction)

processRealizationsStandard = standardEuler.getRealizationsAtGivenTime(maturity)

europeanOptionStandard = SimpleEuropeanOption(processRealizationsStandard, r)

priceStandard = europeanOptionStandard.getPrice(payoffFunction, maturity)

print(priceStandard)