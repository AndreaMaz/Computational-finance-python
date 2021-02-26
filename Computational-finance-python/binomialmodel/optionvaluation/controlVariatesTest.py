#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test how the use of control variates can make the approximation of the
valuation of an american call and put option better.

@author: Andrea Mazzon
"""

import math
from binomialmodel.optionvaluation.controlVariates import AmericanOptionWithControlVariates
import numpy as np 
import matplotlib.pyplot as plt


initialValue = 1
r = 0.02
sigma = 0.7
maturity = 3

strike = initialValue

maximumNumberOfTimes = 150

americanCV = np.empty((maximumNumberOfTimes - 1))
americanBinomial = np.empty((maximumNumberOfTimes - 1))
european = np.empty((maximumNumberOfTimes - 1))


for numberOfTimes in range (2, maximumNumberOfTimes + 1):

    
    interestRate = math.exp(r * maturity / numberOfTimes) - 1
    
    evaluator = AmericanOptionWithControlVariates(initialValue, r, sigma, maturity, strike) 
    
    americanCV[numberOfTimes - 2] = evaluator.getAmericanCallAndPutPriceWithControlVariates(numberOfTimes)[1]
    americanBinomial[numberOfTimes - 2] = evaluator.getAmericanCallAndPutPriceWithBinomialModel(numberOfTimes)[1]
    european[numberOfTimes - 2] = evaluator.getEuropeanCallAndPutPriceWithBinomialModel(numberOfTimes)[1]
  
blackScholesPrice = evaluator.blackScholesPriceCallAndPut()[1] 
blackScholesVector = np.full((maximumNumberOfTimes - 1), blackScholesPrice)

plt.plot(americanCV)
plt.plot(americanBinomial)
plt.plot(european)
plt.plot(blackScholesVector)
plt.ylim([americanCV[-1]*94/100, americanCV[-1]*104/100])
plt.legend(('american, control variates','american, binomial model',
            'european, binomial model','european, analytic'))
plt.title("Control variates approximation of an American option for a BS model")
plt.show()      