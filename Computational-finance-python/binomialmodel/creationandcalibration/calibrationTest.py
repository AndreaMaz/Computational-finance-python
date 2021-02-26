#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we class the calibration of the parameters u > 1 + rho, where rho is the
interest rate, and d < 1 such that S(j+1)=uS(j) with probability q and
S(j+1)=dS(j) with probability d 

@author: Andrea Mazzon
"""
from binomialmodel.creationandcalibration.binomialModelCalibration import BinomialModelCalibration

initialValue = 100.0
decreaseIfDown = 0.6
increaseIfUp = 2
finalTime = 13
interestRate = 0.1
riskNeutralProbabilityUp = (1 + interestRate - decreaseIfDown)/ (increaseIfUp - decreaseIfDown)

calibrator = BinomialModelCalibration(interestRate, riskNeutralProbabilityUp, initialValue)

calibratedIncreaseIfUp, calibratedDecreaseIfDown = \
    calibrator.testCalibration(decreaseIfDown, increaseIfUp, finalTime)
    
print()    
print("The original parameters are ", increaseIfUp, " and ", decreaseIfDown)
print()
print("The calibrated ones are ", calibratedIncreaseIfUp, " and ", calibratedDecreaseIfDown)
   