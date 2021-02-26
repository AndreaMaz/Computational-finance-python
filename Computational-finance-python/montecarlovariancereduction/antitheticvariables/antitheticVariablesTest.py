#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test the comparison of the average errors we get with standard Monte-Carlo
and Antithetic Variables when valuating an European call option. 

In particular, we plot the two average errors for an incresing number of
simulations.

@author: Andrea Mazzon
"""

from compareStandardMCWithAV import compare

import matplotlib.pyplot as plt

#process parameters
initialValue = 100
r = 0.05
sigma = 0.5

#option parameters
T = 3
strike = initialValue


#we want to test the two methods for different numbers of simulations
numbersSimulations = [10**k for k in range(3,6)]

#the function compare returns a 2-uple. The first value is the average error of the
#standard Monte-Carlo method, so it is the entry of the 2-uple at position 0.
averageErrorsWithStandardMC = [compare(numberOfSimulations, initialValue, r, sigma, T, strike)[0] \
    for numberOfSimulations in numbersSimulations]
    
#The second value returned by compare is the average error of the Monte-Carlo
#method with Antithetic Variables, so it is the entry of the 2-uple at position 1.
averageErrorsWithAV = [compare(numberOfSimulations, initialValue, r, sigma, T, strike)[1] \
    for numberOfSimulations in numbersSimulations]
   
plt.plot(numbersSimulations,averageErrorsWithStandardMC, 'bo')
plt.plot(numbersSimulations,averageErrorsWithAV, 'ro')
plt.xlabel('Number of simulations')
plt.ylabel('Average error')
plt.title('Average errors in the valuation of a call option for standard Monte-Carlo and Antithetic Variables methods')
plt.show() 