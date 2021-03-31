#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import numpy as np
import math
import matplotlib.pyplot as plt


#from gPDESolution import GPDESolution

from gPDESolutionImplicit import GPDESolutionImplicit
from bisectionmethod.bisectionMethod import bisection

class GBrownianMotion:
    """
    It simulates the trajectories of the G-Brownian motion, by approximating its
    cumulative distribution function via finite difference methods to solve
    the related nonlinear PDE and inverting it via a bisection search algorithm.
    
    Attributes
    ----------
    dxFirstDistributions : float
        the length of the sub-intervals in the discretization of the interval
        on which we want to compute the cumulative distribution function by
        solving the PDE
    spaceDiscretizationFirstDistributions : array
        the discretization of the interval on which we want to compute the
        cumulative distribution function by solving the PDE
    dtGBrownianIncrements : float
        the length of the sub-intervals on which we want
        to simulate the G-Brownian increments. Not to be confused with the length
        of the time intervals in the discretization for the finite difference
        methods for solving the PDE
    numberOfTimes : int
        the number of times in the time discretization where we simulate the
        G-Brownian motion
    numberOfSimulations : int
        the number of simulated trajectories of the G-Brownian motion
    pdeSolver : gbrownianmotiongeneration 
        the object that we use to solve the PDE associated to the cumulative
        distributionfunction of the G-Brownian motion            

    Methods
    -------
    getSolutionAtNextTime():
        It returns the solution at the next time step
    solveAndPlot():
        It solves the PDE and dynamically plots the solution at every time step
        of length 0.1. It does not store the solution in a matrix
    solveAndSave():
        It solves the PDE and store the solution as a matrix in the self.solution
        attribute of the class. It also returns it.
    getSolutionForGivenMaturityAndValue(time, space):
        It returns the solution at given time and given space
    
    """
    
    def __init__(self, dx, dt, xmin, xmax, tmax, minusA, plusA, dtGBrownianIncrements, 
                 dxFirstDistributions, sigmaDown, sigmaUp, numberOfSimulations):

        self.pdeSolver = GPDESolutionImplicit(dx, dt, xmin, xmax, tmax, sigmaDown,#min vol of increments 
            sigmaUp)#max vol of increments
        
        self.dxFirstDistributions = dxFirstDistributions
        self.minusA = minusA
        self.plusA = plusA
        #it will be used for the interpolation of the cdf
        self.spaceDiscretizationFirstDistributions = np.arange(minusA, plusA + dxFirstDistributions, dxFirstDistributions) 
        
        #parameters for the simulation of the trajyetories of the G-Brownian
        #motion once we solve the PDE and we interpolate. 
        self.dtGBrownianIncrements = dtGBrownianIncrements
        self.numberOfTimesForGBM = math.ceil(tmax/dtGBrownianIncrements)+1
        self.timeForGBM = np.arange(0, tmax + dtGBrownianIncrements, dtGBrownianIncrements)
        
        self.numberOfSimulations = numberOfSimulations
        
        self.__setFirstDistributions()#the base on which we then interpolate

        print(self.firstDistributions)
        self.__generateGBrownianMotion()#we generate the trajectories once for all
        """
        dt : float
            discretization step related to the time for the numeric solution
            to the PDE associated to the cumulative distribution function of
            the G-Brownian motion
        xmin : float
            left end of the space domain for the numeric solution to the PDE
            associated to the cumulative distribution function of the
            G-Brownian motion. This is also the left end of the interval that
            we discretize to compute the values of the cumulative distribution
            function that constitute the base to interpolate
        xmax : float
            rleft end of the space domain for the numeric solution to the PDE
            associated to the cumulative distribution function of the
            G-Brownian motion. This is also the left end of the interval that
            we discretize to compute the values of the cumulative distribution
            function that constitute the base to interpolate
        tmax : float
            rthe maximum time until which we simulate the G-Brownian motion.
        dtGBrownianIncrements : float
            the length of the sub-intervals on which we want to simulate the
            G-Brownian increments. Not to be confused with the length of the time
            intervals in the discretization for the finite difference methods
            for solving the PDE
        dxFirstDistributions : float
            the length of the sub-intervals in the discretization of the interval
            on which we want to compute the cumulative distribution function by
            solving the PDE
        sigmaDown : float
            the minimum volatility we think our process can have
        sigmaUp : float
            the maximum volatility we think our process can have
        numberOfSimulations : int
            the number of simulated trajectories of the G-Brownian motion   
        """
      
        
    def __getDistributionForGivenThreshold(self, threshold):
        #we don't want to create a new object for every new initial condition:
        #remember that the threshold identifies the initial condition of the PDE
        self.pdeSolver.setThresholdForInitialCondition(threshold)
        #look at the script!
        return self.pdeSolver.getSolutionForGivenTimeAndValue(1, 0)
        
    
    def __setFirstDistributions(self):
        #we set here the values of teh distribution for the discretized times,
        #on which we base to interpolate
        self.firstDistributions = \
            [self.__getDistributionForGivenThreshold(threshold) for threshold in self.spaceDiscretizationFirstDistributions]
            
        
    def __getInterpolatedSolution(self, threshold):
        #searchsorted takes as an input a sorted vector, which the space discretization
        #is, and a value. It returns the position at which the value must be 
        #placed in the vector to keep it ordered. For example, try to type
        #np.searchsorted([1,2,3],1.5)
        thresholdIndex = np.searchsorted(self.spaceDiscretizationFirstDistributions, threshold)
        #the case when the threshold we give is smaller than the left extreme
        #of the interval
        if thresholdIndex == 0:
            return self.firstDistributions[0]
        #the case when the threshold we give is smalbiggerler than the right extreme
        #of the interval
        if thresholdIndex == len(self.spaceDiscretizationFirstDistributions):
            return self.firstDistributions[-1]
        
        #linear interpolation 
        biggerPoint = self.spaceDiscretizationFirstDistributions[thresholdIndex]
        smallerPoint = self.spaceDiscretizationFirstDistributions[thresholdIndex - 1]
        
        biggerDistribution = self.firstDistributions[thresholdIndex]
        smallerDistribution = self.firstDistributions[thresholdIndex - 1]
        
        return ((biggerPoint - threshold) * smallerDistribution + \
                (threshold - smallerPoint) * biggerDistribution) \
            /self.dxFirstDistributions
        
    
    def __findRoot(self, uniformRealization):
        #we invert here the approximated cumulative distribution function via
        #bisection search. This is the function of which we want to find the zero 
        def cumulativeDistributionRemainder(x):
            #if this is zero, it means that F(y) = uniformRealization, so
            #y = F^(-1)(uniformRealization)
            return self.__getInterpolatedSolution(x) - uniformRealization
        #we know that the Brownian increments will quite for sure not be smaller
        #than -2 or bigger than 2
        return bisection(cumulativeDistributionRemainder, self.minusA, self.plusA, 0, 1000)
            
        
    def __generateGBrownianMotion(self):
        
        gBrownianRealizations = np.zeros((self.numberOfTimesForGBM,self.numberOfSimulations))
        #in this way, we can apply it to a vector!
        findRoots = np.vectorize(self.__findRoot)
        
        #the dimension is numberOfTimes - 1 because we need to generate n-1
        #increments to have n realizations 
        uniformRealizations = np.random.uniform(size=(self.numberOfTimesForGBM - 1,self.numberOfSimulations))
        
        for timeIndex in range(1, self.numberOfTimesForGBM):
            
            gBrownianIncrements = findRoots(uniformRealizations[timeIndex-1])\
                * math.sqrt(self.dtGBrownianIncrements)
            
            gBrownianRealizations[timeIndex] = gBrownianRealizations[timeIndex-1] \
                + gBrownianIncrements
        
        self.gBrownianMotionRealizations = gBrownianRealizations
    
    
    def getGBrownianMotion(self):
        """
        It returns the matrix with the realizations of the G-Brownian motion

        Returns
        -------
        array
            the matrix with the realizations of the G-Brownian motion. Row k
            hosts all teh simulated values at time index k.

        """
        
        return self.gBrownianMotionRealizations
        
    
    def getGBrownianMotionAtGivenTimeIndex(self, timeIndex):
        """
        It returns all the simulated values of the G-Brownian motion at a given
        time index.

        Parameters
        ----------
        timeIndex : int
            the time index at which we want to get the simulated values of the
            G-Brownian motion

        Returns
        -------
        array
            the vector hosting the simulated values of the G-Brownian motion
            at the given time index.

        """
        
        return self.gBrownianMotionRealizations[timeIndex]
        
        
    def getGBrownianMotionAtGivenTime(self, time):  
        """
        It returns all the simulated values of the G-Brownian motion at a given
        time.

        Parameters
        ----------
        timeIndex : float
            the time at which we want to get the simulated values of the
            G-Brownian motion

        Returns
        -------
        array
            the vector hosting the simulated values of the G-Brownian motion
            at the given time.

        """
        
        timeIndex = round(time/self.dtGBrownianIncrements)
        
        return self.gBrownianMotionRealizations[timeIndex]
        
    
    def getAverageAtGivenTime(self, time):  
        """
        It returns the average of the simulated values of the G-Brownian motion
        at a given time.

        Parameters
        ----------
        timeIndex : float
            the time at which we want to get the simulated values of the
            G-Brownian motion

        Returns
        -------
        float
            the average of the simulated values of the G-Brownian motion
            at the given time

        """
        
        return np.average(self.getGBrownianMotionAtGivenTime(time))
    
    
    def getPathForGivenSimulation(self, simulationIndex):
        """
        It returns the path of the G-Brownian motion for a given simulation
        index

        Parameters
        ----------
        simulationIndex : int
            the index of the path we want to get

        Returns
        -------
        array
            the vector describing the path simulated

        """
        
        return self.gBrownianMotionRealizations[:, simulationIndex]
    
    
    def plotPaths(self, simulationIndex, numberOfPaths):
        """
        It plots the paths of the process from simulationIndex to 
        simulationIndex + numberOfPaths

        Parameters
        ----------
        simulationIndex : int
           the simulation from which we want to plot
        numberOfPaths : int
            the number of paths we wantto plot

        Returns
        -------
        None.

        """
        for k in range(numberOfPaths):
            path = self.getPathForGivenSimulation(simulationIndex + k);
            self.timeForGBM 
            plt.plot(self.timeForGBM, path)
        plt.xlabel('Time')
        plt.ylabel('Realizations of the process')
        plt.show()           
        
        
        
            
