#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class we test the computation of the price of an european option written
on a binomial model

@author: Andrea Mazzon
"""

import time

from europeanOption import EuropeanOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart



initialValue = 100
decreaseIfDown = 0.5
increaseIfUp = 2
numberOfTimes = 5
interestRate = 0.1

myBinomialModelSmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 

myPayoffEvaluator = EuropeanOption(myBinomialModelSmart)

maturity = numberOfTimes - 1

payoff = lambda x : max(x-initialValue,0)

start = time.time()

priceWithWeightedSum = myPayoffEvaluator.evaluateDiscountedPayoff(payoff, maturity)
priceFromStrategy = myPayoffEvaluator.getInitialDiscountedValuePortfolio(payoff, maturity)

print("The discounted price computed by weighting the payoff with the realizations' probabilities is",
      priceWithWeightedSum)
print()
print("The discounted price of the option computed going backward is ",
      priceFromStrategy)
print()

currentTime = maturity - 1

amountInRiskyAsset, amountInRiskFreeAsset = myPayoffEvaluator.getStrategyAtGivenTime(payoff, currentTime, maturity)

end = time.time()

print("The amount of money that has to be invested in the risky asset at time ",
      currentTime, " is")

print()

print('\n'.join('{:.3}'.format(amount) for amount in amountInRiskyAsset))

print()

print("The amount of money that has to be invested in the risk free asset at time ",
      currentTime, " is")

print()

print('\n'.join('{:.3}'.format(amount) for amount in amountInRiskFreeAsset))

print()

print("Elapse time ", end - start)