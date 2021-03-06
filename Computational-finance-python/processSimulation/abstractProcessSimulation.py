#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AndreaMazzon
"""

import numpy as np
import math
from random import seed

class AbstractProcessSimulation:
    """
    This is an abstract class whose mail goal is to simulate a continuous stochastic
    process. The methods providing drift and diffusion are implemented in 
    sub-classes. 
    
    It is also possible to simulate the process under a given transform, and
    then to transform it back.
    
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

    """
    
    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValue,\
                 transform = lambda x : x, inverseTransform = lambda x : x, mySeed = None):
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

        Returns
        -------
        None.

        """       
        self.numberOfSimulations = numberOfSimulations
        self.timeStep = timeStep
        self.finalTime = finalTime
        self.initialValue = initialValue
        self.transform = transform
        self.inverseTransform = inverseTransform
        self.mySeed = mySeed
        #we generate all the paths for all the simulations
        self.__generateRealizations()
        
        
    

    def __generateRealizations(self):
        
        #look at the function vectorize: we can use it to be able to assign an
        #array to a method, or a function, which is supposed to be defined for
        #floats
        vectorizedGetDrift = np.vectorize(self.getDrift)
        vectorizedGetDiffusion = np.vectorize(self.getDiffusion)
        
        vectorizedTransform = np.vectorize(self.transform)
        #vectorizedInverseTransform = np.vectorize(self.inverseTransform)
        
        numberOfTimes = round(self.finalTime/self.timeStep) + 1
        
        #times on the rows
        self.realizations = np.zeros((numberOfTimes,self.numberOfSimulations)) 
        self.realizations[0] = [self.inverseTransform(self.initialValue)]*self.numberOfSimulations
        
        seed(self.mySeed)

        standardNormalRealizations = np.random.standard_normal((numberOfTimes, self.numberOfSimulations))
        
        #possibly used in order to get the drift and the diffusion
        currentTime = self.timeStep  
        for timeIndex in range(1, numberOfTimes):
            pastRealizations = self.realizations[timeIndex - 1]
            
            
            self.realizations[timeIndex] = self.realizations[timeIndex - 1] \
                + self.timeStep * vectorizedGetDrift(currentTime, pastRealizations)\
                    + vectorizedGetDiffusion(currentTime, pastRealizations) \
                        *  math.sqrt(self.timeStep) * standardNormalRealizations[timeIndex] #the Brownian motion
                        
        self.realizations = vectorizedTransform(self.realizations)
                        
        
    def getRealizations(self):
        """
        It returns all the realizations of the process

        Returns
        -------
        array
            matrix containing the process realizations. The n-th row contains 
            the realizations at time t_n

        """
        return self.realizations
    
    def getRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the realizations of the process at a given time index

        Parameters
        ----------
        timeIndex : int
             the time index, i.e., the row of the matrix of realizations. 

        Returns
        -------
        array
            the vector of the realizations at given time index

        """
        
        return self.realizations[timeIndex]            
                        
    def getRealizationsAtGivenTime(self, time):
        """
        It returns the realizations of the process at a given time

        Parameters
        ----------
        time : float
             the time at which the realizations are returned 

        Returns
        -------
        array
            the vector of the realizations at given time

        """
        
        indexForTime = round(time/self.timeStep)
        return self.realizations[indexForTime]
    
    def getAverageRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the average realizations of the process at a given time index

        Parameters
        ----------
        time : int
             the time index, i.e., the row of the matrix of realizations.

        Returns
        -------
        float
            the average of the realizations at given time index

        """
        
        return np.average(self.realizations(timeIndex))
    
    def getAverageRealizationsAtGivenTime(self, time):
        """
        It returns the average realizations of the process at a given time 

        Parameters
        ----------
        time : int
             the time at which the realizations are returned 

        Returns
        -------
        float
            the average of the realizations at given time

        """
        
        return np.average(self.getRealizationsAtGivenTime(time))
         