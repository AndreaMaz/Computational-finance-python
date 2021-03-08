#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class we test the computation of the price of a knock-out option written
on a binomial model

@author: Andrea Mazzon
"""

#from europeanOption import EuropeanOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart
from knockOutOption import KnockOutOption


initialValue = 100
decreaseIfDown = 0.8
increaseIfUp = 1.2
numberOfTimes = 5
interestRate = 0.0

myBinomialModelSmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 

myPayoffEvaluator = KnockOutOption(myBinomialModelSmart)

maturity = numberOfTimes - 1

payoff = lambda x : max(x-initialValue,0)

lowerBarrier = 75
upperBarrier = 150

processRealizations = myBinomialModelSmart.getRealizationsAtGivenTime(maturity - 1)

priceFromStrategy = myPayoffEvaluator.getInitialDiscountedValuePortfolio(payoff, maturity, lowerBarrier, upperBarrier)

print("The discounted price of the option computed going backward is ",
      priceFromStrategy)

valuesOption = myPayoffEvaluator.getValuesPortfolioBackward(payoff, maturity, lowerBarrier, upperBarrier)
realizationOfTheProcess = myBinomialModelSmart.getRealizations()