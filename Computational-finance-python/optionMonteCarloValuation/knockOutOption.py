#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import numpy as np
import math

class KnockOutOption:
    """
    This class provides a valuation of a Knock-out barrier option.
    
    It has a method that takes a matrix or realizations of the process. The
    columns represent the paths of the process. It returns a vector whose k-th
    realization is equal to the payoff function of the value of the k.th
    path at maturity if the path up to maturity lies within an interval, and
    zero otherwise.
    
    Then another method computes the average of the vector returned by the
    method above.
    
    We use this class within an example about Antithetic variables.

    ...

    Attributes
    ----------
    payoffFunction : function
        the function representing the payoff of the option at maturity
    maturity : float
        the maturity of the option
    lowerBarrier: float
        the lower barrier of the option. Default value -infinity
    upperBarrier: float
        the upper barrier of the option. Default value +infinity
    r : float
        interest rate
        

    Methods
    -------
    getPayoff(processRealizations)
        It returns the vector of the payoff of the option for all the simulations
    getPrice(payoffFunction):
        It returns the price of the option
    printPrice(payoffFunction):
        It prints the price of the option t
    """
     #Python specific syntax for the constructor
    def __init__(self, payoffFunction, maturity, lowerBarrier = -np.infty, upperBarrier = np.infty,  r = 0):#r = 0 if not specified
        """    
        payoffFunction : function
            the function representing the payoff of the option at maturity
        maturity : float
            the maturity of the option
        lowerBarrier: float
            the lower barrier of the option. Default value -infinity
        upperBarrier: float
            the upper barrier of the option. Default value +infinity
        r : float
            interest rate
        """
        self.payoffFunction = payoffFunction
        self.maturity = maturity
        self.lowerBarrier = lowerBarrier
        self.upperBarrier = upperBarrier
        self.r = r
        
    
    
    def getPayoff(self, processRealizations):
        """
        It returns the vector of the payoff of the option for all the simulations

        Parameters
        ----------
        processRealizations : array
            the matrix representing the realizations of the process. The
            columns represent the paths of the process.

        Returns
        -------
        payoffRealizations : array
            a vector whose k-th realization is equal to the payoff function of
            the value of the k.th path at maturity if the path up to maturity
            lies within an interval, and zero otherwise.

        """
        #np.amax returns the maximum value of an array.
        #processRealizations[-1,k] is the realization of the process at maturity
        #for simulation k, if we assume that the process is simulated up to 
        #maturity. Otherwise, we can get the index corresponding to the maturity
        #if we know the time step of the process.
        payoffRealizations = [self.payoffFunction(processRealizations[-1,k]) \
                              if np.amax(processRealizations[:,k]) < self.upperBarrier \
                                  and np.amin(processRealizations[:,k]) > self.lowerBarrier \
                               else 0 for k in range(len(processRealizations[0]))]
        
        
        return payoffRealizations
    
    
    def getPrice(self, processRealizations):
        """
        It returns the discounted price of the option defined by payoffFunction
        and payed at maturity. 
        
        The underlying at maturity is defined by the realizations of the process
        stored in self.processRealizations.

        Parameters
        ----------
        payoffFunction : function
            the function describing the payoff of the option
        maturity : double
            the maturity of the option

        Returns
        -------
        float
            the price of the option.

        """
        payoffRealizations = self.getPayoff(processRealizations)
        #look at the use of numpy.mean: we get the average of the elements
        #of a list
        return math.exp(-self.r * self.maturity) * np.mean(payoffRealizations)
    
    
    def printPrice(self, processRealizations):
        """
        It prints the discounted price of the option defined by payoffFunction
        and payed at maturity. 
        
        The underlying at maturity is defined by the realizations of the process
        stored in self.processRealizations.

        Parameters
        ----------
        payoffFunction : function
            the function describing the payoff of the option
        maturity : double
           the maturity of the option 

        Returns
        -------
        None

        """
        
        price = self.getPrice(processRealizations)
        #we want to print 3 decimal digits
        print('The discounted price of the option is {:.3}'.format(price))
    
