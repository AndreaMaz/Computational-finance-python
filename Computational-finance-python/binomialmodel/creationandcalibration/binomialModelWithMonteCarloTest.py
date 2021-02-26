#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test the simulation of the binomial model with the pure Monte Carlo
approach. We print a single path of the process, i.e., for a given simulation/
state of the world. We also print and plot the evolution of the discounted average
and of the probability that the discounted value of a future realization of the
process is bigger than the initial value.

@author: Andrea Mazzon
"""
#look how imports work
from binomialModelMonteCarlo import BinomialModelMonteCarlo

    
initialValue = 3
decreaseIfDown = 0.5
increaseIfUp = 2
numberOfTimes = 15
numberOfSimulations = 100
interestRate = 0

#look how to construct an object in Python
myBinomialModel = BinomialModelMonteCarlo(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, numberOfSimulations, interestRate)    

#we print the 10-th path of the process
simulationNumber = 8;
myBinomialModel.printPath(simulationNumber)

myBinomialModel.plotPath(simulationNumber, 2)
#prints..
myBinomialModel.printEvolutionDiscountedAverage()
myBinomialModel.printEvolutionProbabilitiesOfGain()

#..and plots
myBinomialModel.plotEvolutionDiscountedAverage()
myBinomialModel.plotEvolutionProbabilitiesOfGain()
