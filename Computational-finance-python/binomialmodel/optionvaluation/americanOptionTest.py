#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class we test the computation of the price of an american call 
option written on a binomial model

@author: Andrea Mazzon
"""

from americanOption import AmericanOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart


initialValue = 20
decreaseIfDown = 0.9
increaseIfUp = 1.1
numberOfTimes = 4
interestRate = 0.05

myBinomialModelSmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 

myPayoffEvaluator = AmericanOption(myBinomialModelSmart)

maturity = numberOfTimes - 1

payoff = lambda x : max(initialValue-x,0)

valuesOption, valuesExercise, valuesIfWait, exercise = \
    myPayoffEvaluator.analysisOption(payoff, maturity)

print("The value of the option is ", myPayoffEvaluator.getValueOption(payoff, maturity))
