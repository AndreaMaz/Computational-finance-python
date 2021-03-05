#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import numpy as np
import math

class GenerateBlackScholes:
    """
    In this class we generate N realizations of a log-normal process
    dX_t = r X_t dt + sigma X_t dW_t
    at time T>0.
    
    We do it by writing
    X_T = X_0 exp((r- 0.5 sigma^2) T + sigma T^0.5 Z),
    where Z is a standard normal random variable.
    
    We proceed in two different ways:
        - we generate the N realizations of Z directly;
        - we first generate N/2 realizations of Z, and then set 
        Z(j+N/2) = - Z(j).
        
    The second way accounts for the Antithetic Variables variance reduction method.
    
    
    Attributes
    ----------
    numberOfSimulations : int
        the number of simulated values of the process at maturity
    T : float
        the maturity of the option
    initialValue : float
        the initial value of the process
    sigma : float
        the standard deviation
    r : float
        the interest rate. Default = 0  
    seed : int
        the seed to generate the sequence both with standard Monte Carlo and
        Antithetic variables. Default = None 

    Methods
    -------
    generateRealizations(self):
        It generates a number N = self.numberOfSimulations of realizations of the
        log-normal process at time self.T.
    generateRealizationsAntitheticVariables(self):
        It generates a number N = self.numberOfSimulations of realizations of the
        log-normal process at time self.T, using Antithetic Variables
    """
    
     #Python specific syntax for the constructor
    def __init__(self, numberOfSimulations, T, initialValue, sigma, r = 0,#r = 0 if not specified
                 seed = None):#no seed if not specified
        """    
        Parameters
        ----------
        numberOfSimulations : int
           the number of simulated values of the process at maturity
        T : float
            the maturity of the option
        initialValue : float
            the initial value of the process
        sigma : float
            the standard deviation
        r : float
            the interest rate. Default = 0
        seed : int
            the seed to generate the sequence both with standard Monte Carlo and
            Antithetic variables. Default = None 
        """
        self.numberOfSimulations = numberOfSimulations
        self.T = T
        self.initialValue = initialValue
        self.sigma = sigma
        self.r = r
        #we give the seed for the random number generator. Note that if we specify
        #no seed at all, self.seed will have value None, that is, no value (like
        #Null in Java). In this case, it will call np.random.seed(). Indeed,
        #the value of the seed is a default argument in np.random.seed
        np.random.seed(seed)
        
        
    def generateRealizations(self):
        """
        It generates a number N = self.numberOfSimulations of realizations of the
        log-normal process at time self.T.
        
        In particular, it does it by generating N values of standard normal
        random variables Z(j), j = 1, ..., N, and computing for every j
        X_T(j) = X_0 exp((r- 0.5 sigma^2) T + sigma T^0.5 Z(j))

        Returns
        -------
        blackScholesRealizations : list
            a vector representing the realizations of the process

        """
                    
        #note the way to get a given number of realizations of a standard normal
        #random variable. Also note that in order to access the specific field of the
        #the class, we have to refer to it with "self.". Same things for methods
        standardNormalRealizations = np.random.standard_normal(self.numberOfSimulations)

        #we don't want to compute this every time.
        firstPart = self.initialValue * math.exp((self.r - 0.5 * self.sigma**2) * self.T)
        
        def BSFunction(x):
            return firstPart * math.exp(self.sigma * math.sqrt(self.T) * x)
        
        #look at this peculiar Python for loop: this is equivalent to write
        #for k in range (standardNormalRealizations.length)
        #    blackScholesRealizations[k] = firstPart * math.exp(self.sigma * math.sqrt(self.T) * blackScholesRealizations[k])
        #The part (fox x in self.processRealizations) is similar to the Java foreach.
        #loop. 
        blackScholesRealizations = [BSFunction(x) for x in standardNormalRealizations]
            
        return blackScholesRealizations
       
        
    def generateRealizationsAntitheticVariables(self):
        """
        It generates a number N = self.numberOfSimulations of realizations of the
        log-normal process at time self.T.
        
        In particular, it does it by first generating N values of standard normal
        random variables Z(j), j = 1, ..., N/2, Z(n/2+j)=-Z(j), j = 1, ..., N/2
        and computing for every j
        X_T(j) = X_0 exp((r- 0.5 sigma^2) T + sigma T^0.5 Z(j)).
         
        If N is odd, N/2 is defined as the smallest integer >= N/2

        Returns
        -------
        blackScholesRealizations : list
            a vector representing the realizations of the process

        """
        
        
        #math.ceil(x) returns the smallest integer >= x
        standardNormalRealizations = np.random.standard_normal(math.ceil(self.numberOfSimulations/2))
        
        #we don't want to compute this every time.
        firstPart = self.initialValue * math.exp((self.r - 0.5 * self.sigma**2) * self.T)
        
        #note the use of the concatenation operator "+" between Python lists 
        blackScholesRealizations = [firstPart * math.exp(self.sigma * math.sqrt(self.T) * x) \
            for x in standardNormalRealizations] + \
            [firstPart * math.exp(self.sigma * math.sqrt(self.T) * (-x)) \
            for x in standardNormalRealizations]
               
        return blackScholesRealizations