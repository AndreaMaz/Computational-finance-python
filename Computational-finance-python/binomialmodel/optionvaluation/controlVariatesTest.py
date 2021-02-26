#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test how the use of control variates can make the approximation of the
valuation of an american call and put option better.

@author: Andrea Mazzon
"""

from controlVariates import AmericanOptionWithControlVariates
import matplotlib.pyplot as plt

#parameters for the model
initialValue = 1
r = 0.02
sigma = 0.7

#paramteres for the option
maturity = 3
strike = initialValue


maximumNumberOfTimes = 150

americanCV = []
americanBinomial =[]
european = []


for numberOfTimes in range (2, maximumNumberOfTimes + 1):
    
    evaluator = AmericanOptionWithControlVariates(initialValue, r, sigma, maturity, strike) 
    
    americanCV.append(evaluator.getAmericanCallAndPutPriceWithControlVariates(numberOfTimes)[1])
    americanBinomial.append(evaluator.getAmericanCallAndPutPriceWithBinomialModel(numberOfTimes)[1])
    european.append(evaluator.getEuropeanCallAndPutPriceWithBinomialModel(numberOfTimes)[1])
  
blackScholesPrice = evaluator.blackScholesPriceCallAndPut()[1] 
blackScholesVector = [blackScholesPrice] * (maximumNumberOfTimes - 1)

plt.plot(americanCV)
plt.plot(americanBinomial)
plt.plot(european)
plt.plot(blackScholesVector)
plt.ylim([americanCV[-1]*94/100, americanCV[-1]*104/100])
plt.legend(('american, control variates','american, binomial model',
            'european, binomial model','european, analytic'))
plt.title("Control variates approximation of an American option for a BS model")
plt.show()      