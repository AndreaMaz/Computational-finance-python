#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we want to approximate the price of an American put option with maturity T
and strike K, written on an underlying following the Black-Scholes model (i.e., 
having log-normal dynamics) with log-volatility sigma and risk-freee rate r. It
can be seen that such a model can be approximated by a binomial model with N
times, for N large enough, u = exp(sigma*sqrt(T/N)), d=1/u, rho = exp(rT/N)-1. 
We then valuate the option under such a binomial model and hope that the price
converges for large N.

@author: Andrea Mazzon
"""
import math
import matplotlib.pyplot as plt

from americanOption import AmericanOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart


initialValue = 1
r = 0.02
sigma = 0.7
maturity = 3

#we are at the money
payoff = lambda x : max(initialValue - x,0)


#the maximum number of times N we use to approximate the price
maximumNumberOfTimes = 150

#we want to keep track of the prices and plot them
prices = []

#we also want to keep track of how u and d evolve for increasing N
increaseIfUps = []
decreaseIfDowns = []

for numberOfTimes in range (2, maximumNumberOfTimes + 1):
    
    increaseIfUp = math.exp(sigma * math.sqrt(maturity / numberOfTimes))
    decreaseIfDown = 1/increaseIfUp
    
    #we keep track of how u and d evolve for increasing N
    increaseIfUps.append(increaseIfUp)
    decreaseIfDowns.append(decreaseIfDown)
    
    interestRate = math.exp(r * maturity / numberOfTimes) - 1
    
    binomialmodel = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 
    
    
    myPayoffEvaluator = AmericanOption(binomialmodel)
    prices.append(myPayoffEvaluator.getValueOption(payoff, numberOfTimes - 1))
    
  
plt.plot(prices)
plt.xlabel('Number of time steps')
plt.ylabel('Price')
plt.title("Price of an American option for a BS model, approximated via binomial model")
plt.ylim([prices[-1]*97/100, prices[-1]*102/100])
plt.show()      