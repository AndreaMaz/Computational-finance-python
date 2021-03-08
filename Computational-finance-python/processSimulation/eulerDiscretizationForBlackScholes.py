#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import math

from processSimulation.generalProcessSimulation import GeneralProcessSimulation

class EulerDiscretizationForBlackScholes(GeneralProcessSimulation):
    """
    It provides the simulation of a Black-Scholes process, by simulating its 
    logarithm. In this way we don't have discretization error
    
    Attributes
    ----------
    numberOfSimulations : int
        the number of simulated paths.
    timeStep : float
        the time step of the time discretization.
    finalTime : float
        the final time of the time discretization.
    initialValue : float
        the initial value of the process.
    mySeed : int, optional
        the seed to the generation of the standard normal realizations
    mu: float
        the drift
    sigma: float
        the log-normal volatility
        
    Methods
    ----------
    getRealizations():
        It returns all the realizations of the process
    getRealizationsAtGivenTimeIndex(timeIndex):
        It returns the realizations of the process at a given time index
    getAverageRealizationsAtGivenTimeIndex(timeIndex):
        It returns the average realizations of the process at a given time index
    getAverageRealizationsAtGivenTime(time):
        It returns the average realizations of the process at a given time 
    getDrift(time, realizations)
        It returns the drift of the logarithm of the Black-Scholes process
    getDiffusion(time, realizations)
        It returns the diffusion of the logarithm of the Black-Scholes process
    
    """
    
    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValue,mu, sigma, 
                 mySeed = None): 
        
        self.mu = mu
        self.sigma = sigma
        
        super().__init__(numberOfSimulations, timeStep, finalTime, initialValue, 
                 lambda x : math.exp(x), lambda x : math.log(x), mySeed)
        
    def getDrift(self, time, realizations):
        """
        It returns the drift of the logarithm of the Black-Scholes process 

        Parameters
        ----------
        time : double
            the time. Not used here
        realizations : double
            the realizations of the process. Not used here

        Returns
        -------
        float
            the drift of the logarithm.
        """
        return self.mu - 0.5 * self.sigma**2
    
    
    def getDiffusion(self, time, realizations):
        """
        It returns the diffusion of the logarithm of the Black-Scholes process 

        Parameters
        ----------
        time : double
            the time. Not used here
        realizations : double
            the realizations of the process. Not used here

        Returns
        -------
        float
            the diffusion of the logarithm.
        """
        return self.sigma