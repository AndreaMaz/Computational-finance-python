#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class we do some first tests for the use of Monte-Carlo methods to 
simulate continuous time stochastic processes.

@author: Andrea Mazzon
"""
import numpy as np

from eulerDiscretizationForBlackScholes import EulerDiscretizationForBlackScholes
from standardEulerDiscretization import StandardEulerDiscretization
from optionMonteCarloValuation.simpleEuropeanOption import SimpleEuropeanOption
from analyticformulas.analyticFormulas import blackScholesPriceCall


numberOfSimulations = 10000
timeStep = 0.1
finalTime = 3

initialValue = 2
r = 0.0
sigma = 0.5 


maturity = finalTime
strike = 1


muFunction = lambda t, x : r * x
sigmaFunction = lambda t, x : sigma * x

payoffFunction = lambda x : max(x-strike,0)

analyticPrice = blackScholesPriceCall(initialValue, r, sigma, maturity, strike)

transformErrors = []
standardErrors = []

for k in range(30):

    #price and error generating the process by simulating the logarithm 
    
    eulerBlackScholes= EulerDiscretizationForBlackScholes(numberOfSimulations, timeStep, finalTime, 
                       initialValue, r, sigma)
    
    processRealizationsWithTransform = eulerBlackScholes.getRealizationsAtGivenTime(maturity)
    
    europeanOptionWithTransform = SimpleEuropeanOption(processRealizationsWithTransform, r)
    
    priceWithTransform = europeanOptionWithTransform.getPrice(payoffFunction, maturity)
    
    transformErrors.append(abs(priceWithTransform - analyticPrice)/analyticPrice)
    
    #price generating the process by using the standard Euler scheme 
    
    standardEuler = StandardEulerDiscretization(numberOfSimulations, timeStep, finalTime, 
                       initialValue, muFunction, sigmaFunction)
    
    processRealizationsStandard = standardEuler.getRealizationsAtGivenTime(maturity)
    
    europeanOptionStandard = SimpleEuropeanOption(processRealizationsStandard, r)
    
    priceStandard = europeanOptionStandard.getPrice(payoffFunction, maturity)
    
    standardErrors.append(abs(priceStandard - analyticPrice)/analyticPrice)


print("Average error simulating the logarithm: ", np.mean(transformErrors))
print()
print("Average error using standard Euler scheme: ", np.mean(standardErrors))