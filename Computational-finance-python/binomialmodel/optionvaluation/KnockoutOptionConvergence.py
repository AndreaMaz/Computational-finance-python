#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we want to approximate the price of a Knock-out, down-and-out call option
with maturity T and strike K, written on an underlying following the Black-Scholes
model (i.e., having log-normal dynamics) with log-volatility sigma and risk-free
rate r. It can be seen that such a model can be approximated by a binomial
model with N times, for N large enough, u = exp(sigma*sqrt(T/N)), d=1/u,
rho = exp(rT/N)-1. We then valuate the option under such a binomial model and
hope that the price converges for large N.

@author: Andrea Mazzon
"""
import math
import matplotlib.pyplot as plt

from knockOutOption import KnockOutOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart
from analyticformulas.analyticFormulas import blackScholesDownAndOut

initialValue = 2
r = 0.02
sigma = 0.7
strike = 1.6
lowerBarrier = 1.5

maturity = 3

payoff = lambda x : max (x-strike,0)




#the maximum number of times N we use to approximate the price
maximumNumberOfTimes = 1000

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

    
    myBinomialModelSmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 
    
    myPayoffEvaluator = KnockOutOption(myBinomialModelSmart)
    
    prices.append(myPayoffEvaluator.getInitialDiscountedValuePortfolio(payoff, numberOfTimes-1, lowerBarrier))
    
  
blackScholesPrice = blackScholesDownAndOut(initialValue, r, sigma, maturity, strike, lowerBarrier)
#blackScholesPrice = blackScholesPriceCall(initialValue, r, sigma, maturity, strike)
plt.plot(prices)
blackScholesVector = [blackScholesPrice] * (maximumNumberOfTimes - 1)
plt.plot(blackScholesVector)
plt.xlabel('Number of time steps')
plt.ylabel('Price')
plt.legend(('binomial model','analytic price'))
plt.show()    