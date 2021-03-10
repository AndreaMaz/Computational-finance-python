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
decreaseIfDown = 0.5
increaseIfUp = 2
interestRate = 0.0
numberOfTimes = 5

myBinomialModelSmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 

myPayoffEvaluator = KnockOutOption(myBinomialModelSmart)

maturity = numberOfTimes - 1

payoff = lambda x : max(x-1,0)

lowerBarrier = 75
upperBarrier = 150

#processRealizationsAtMaturity = myBinomialModelSmart.getRealizationsAtGivenTime(maturity - 1)

priceOfTheOption = myPayoffEvaluator.getInitialDiscountedValuePortfolio(payoff, maturity, lowerBarrier, upperBarrier)

print("The discounted price of the option computed going backward is ",
      priceOfTheOption)

valuesOption = myPayoffEvaluator.getValuesPortfolioBackward(payoff, maturity, lowerBarrier, upperBarrier)
processRealizations = myBinomialModelSmart.getRealizations()