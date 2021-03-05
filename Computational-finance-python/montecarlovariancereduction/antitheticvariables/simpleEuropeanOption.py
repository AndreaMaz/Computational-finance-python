#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
from numpy import mean
import math

class SimpleEuropeanOption:
    """
    This class provides a very naive valuation of an European option.
    
    It has a method that takes a list or realizations and applies the payoff
    function to the list
    
    Then another method computes the average of the vector returned by the
    method above.
    
    We use this class within an example about Antithetic variables.

    ...

    Attributes
    ----------
    processRealizations : list
        vector reresenting the realizations of the process at maturity
    r : float
        interest rate
        

    Methods
    -------
    getPayoff(payoffFunction)
        It returns the vector payoffFunction(self.processRealizations)
    getPrice(payoffFunction):
        It returns the price of the option defined by payoffFunction, valuated
        in the realizations of the process stored in self.processRealizations.
    printPrice(payoffFunction):
        It prints the price of the option defined by payoffFunction, valuated
        in the realizations of the process stored in self.processRealizations.t
    """
     #Python specific syntax for the constructor
    def __init__(self, processRealizations, r = 0):#r = 0 if not specified
        """    
        Parameters
        ----------
        processRealizations : list
            a vector representing the realizations of the process at maturity
        r : float
            the interest rate
        """
        self.processRealizations = processRealizations
        self.r = r
    
    
    def getPayoff(self, payoffFunction):
        """
        It returns the vector payoffFunction(self.processRealizations)

        Parameters
        ----------
        payoffFunction : function
            the function describing the payoff of the option

        Returns
        -------
        payoffRealizations : list
            the realizations of the payoff

        """
        #processRealizations[i] -> payoffFunction(processRealizations[i])
        #look at this peculiar Python for loop: this is equivalent to write
        #for k in range (self.processRealizations.length) K=0,1,2,...,self.processRealizations.length-1
        #    payoffRealizations[k] = payoffFunction(self.processRealizations[k])
        #The part (fox x in self.processRealizations) is similar to the Java foreach.
        #loop. 
        payoffRealizations = [payoffFunction(x) for x in self.processRealizations]
        return payoffRealizations
    
    
    def getPrice(self, payoffFunction, maturity):
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
        payoffRealizations = self.getPayoff(payoffFunction)
        #look at the use of numpy.mean: we get the average of the elements
        #of a list
        return math.exp(-self.r * maturity) * mean(payoffRealizations)
    
    
    def printPrice(self, payoffFunction, maturity):
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
        
        price = self.getPrice(payoffFunction, maturity)
        #we want to print 3 decimal digits
        print('The discounted price of the option is {:.3}'.format(price))
    
