#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Here we test the simulation of the binomial model when we 
we don't rely on pseudo random numbers generations, but at a given time we
just consider ALL the possible realizations of the process attributing a
(analytic!) probability to every realization. In particular, we print and plot
the evolution of the discounted average of the process and of the probability
that the discounted value of a future realization of the process is bigger than
the initial value.

@author: Andrea Mazzon
"""
from LogBinomialSmart import LogBinomialModelSmart
from binomialmodel.optionvaluation.europeanOption import EuropeanOption

initialValue = 2.0
decreaseIfDown = 0.5
increaseIfUp = 1.5
numberOfTimes = 150

myBinomialModelSmart = LogBinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes) 

#prints..
myBinomialModelSmart.printEvolutionProbabilitiesOfGain()
myBinomialModelSmart.printEvolutionDiscountedAverage()

#..and plots
myBinomialModelSmart.plotEvolutionProbabilitiesOfGain()
myBinomialModelSmart.plotEvolutionDiscountedAverage()

real = myBinomialModelSmart.getRealizations()