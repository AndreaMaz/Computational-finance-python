#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is devoted to compare the gain we obtain when using Monte-Carlo
with Antithetic variables in the valuation of an European call option written
on a Black-Scholes model.
    
@author: Andrea Mazzon
"""
import numpy as np
from statistics import mean

from generateBlackScholes import GenerateBlackScholes
from simpleEuropeanOption import SimpleEuropeanOption
from analyticformulas.analyticFormulas import blackScholesPriceCall
        
    
def compare(numberOfSimulations, initialValue, sigma, T, strike, r = 0):
    """
    It returns the average errors in the valuation of the call option we get
    using the standard Monte-Carlo method and the Monte-Carlo method with
    Antithetic Variables, respectively, over 100 tests.

    Parameters
    ----------
    numberOfSimulations : int
        the number of simulations of the the values of the process at maturity.
    initialValue : float
        the initial value of the process
    r : float
        the risk free rate
    sigma : float
        the log-volatility
    T : float
        the maturity of the option.
    strike : float
        the strike of the option.

    Returns
    -------
    averageErrorStandardMC : float
        the average error that we get by using the standard Monte-Carlo
        method over 100 tests.
    averageErrorAV : float
        the average error that we get by using the Monte-Carlo method with
        Antithetic Variables over 100 tests

    """
        
    numberOfTests = 100
    
    
    #the two lists that will contain our average errors for the different tests
    errorsStandardMonteCarlo = []
    errorsMonteCarloWithAV = []
    
    #our benchmark: the analytic price of the call option
    analyticPriceBS = blackScholesPriceCall(initialValue, r, sigma, T, initialValue)

    #look at how lambda functions are defined in Python 
    payoff = lambda x : max(x - initialValue, 0)
    
    #alternatively:
    #def payoff(x):
    #    return max(x - initialValue, 0)
    
    #note how to construct an object of a class
            
    blackScholesGenerator = GenerateBlackScholes(numberOfSimulations, T, initialValue, sigma, r)
    #k=0,..,numberOfTests - 1
    for k in range(numberOfTests):
    
            

        #first, the valuation with the standard Monte-Carlo:
        realizationsOfTheProcessWithStandardMC = blackScholesGenerator.generateRealizations()
    
        priceCalculatorWithStandardMC = SimpleEuropeanOption(realizationsOfTheProcessWithStandardMC, r)
        priceStandardMC = priceCalculatorWithStandardMC.getPrice(payoff, T) 
        errorWithStandardMC= abs(priceStandardMC - analyticPriceBS)/analyticPriceBS
        #note how to append new values to a list
        errorsStandardMonteCarlo.append(errorWithStandardMC)
        
        #then, the one with Antithetic Variables:
        realizationsOfTheProcessWithAV = blackScholesGenerator.generateRealizationsAntitheticVariables()
        
        priceCalculatorWithAV = SimpleEuropeanOption(realizationsOfTheProcessWithAV, r)
        priceWithAV = priceCalculatorWithAV.getPrice(payoff, T) 
        errorWithAV= abs(priceWithAV - analyticPriceBS)/analyticPriceBS
        #note how to append new values to a list
        errorsMonteCarloWithAV.append(errorWithAV)

    #we get and return the respective average errors: mean is imported from
    #statistics
    averageErrorStandardMC = mean(errorsStandardMonteCarlo)
    #you can also use numpy
    averageErrorAV = np.mean(errorsMonteCarloWithAV)
    
    return averageErrorStandardMC, averageErrorAV
