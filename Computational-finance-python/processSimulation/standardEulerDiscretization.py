#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:11:13 2021

@author: andreamazzon
"""

from processSimulation.generalProcessSimulation import GeneralProcessSimulation

class StandardEulerDiscretization(GeneralProcessSimulation):
    """
    It provides the simulation of a local volatility process.
    
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
    transform : function, optional
        the function that is applied to simulate the process. 
        The default is the identity.
    inverseTransform : function, optional
        the inverse function that is applied to simulate the process. 
        The default is the identity.
    mySeed : int, optional
        the seed to the generation of the standard normal realizations
    muFunction: float
        the function for the drift
    sigmaFunction: float
        the function for the volatility
        
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
    
    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValue,
                 muFunction, sigmaFunction, transform = lambda x : x, 
                 inverseTransform = lambda x : x, mySeed = None): 
        """
        Parameters
        ----------
        numberOfSimulations : int
            the number of simulated paths.
        timeStep : float
            the time step of the time discretization.
        finalTime : float
            the final time of the time discretization.
        initialValue : float
            the initial value of the process.
        transform : function, optional
            the function that is applied to simulate the process. 
            The default is the identity.
        inverseTransform : function, optional
            the inverse function that is applied to simulate the process. 
            The default is the identity.
        mySeed : int, optional
            the seed to the generation of the standard normal realizations
        muFunction: float
            the function for the drift
        sigmaFunction: float
            the function for the volatility
        """
        
        self.muFunction = muFunction
        self.sigmaFunction = sigmaFunction
        
        super().__init__(numberOfSimulations, timeStep, finalTime, initialValue, 
                 transform, inverseTransform, mySeed)
        
    def getDrift(self, time, realization):
        """
        It returns the drift of the  process 

        Parameters
        ----------
        time : double
            the time.
        realizations : double
            the realizations of the process.

        Returns
        -------
        float
            the drift of the process.
        """
        return self.muFunction(time, realization)
    
    
    def getDiffusion(self, time, realization):
        """
        It returns the diffusion of the  process 

        Parameters
        ----------
        time : double
            the time.
        realizations : double
            the realizations of the process.

        Returns
        -------
        float
            the diffusion of the process.
        """
        return self.sigmaFunction(time, realization)