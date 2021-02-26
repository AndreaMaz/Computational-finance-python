#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Andrea Mazzon
"""

import numpy as np 
from random import seed
import matplotlib.pyplot as plt
import math

from binomialModel import BinomialModel 

#note the syntax to tell the compiler that this class extends the (abstract)
#class BinomialModel
class BinomialModelMonteCarlo(BinomialModel):
    """
    In this class we implement the simulation of a binomial model by using 
    a pure Monte Carlo approach.
    
    This means that we simulate a given number of "states of the world" by 
    producing a sequence of (pseudo) random numbers. In particular, at time j
    and for the simulation i, we generate with the help of Python a (pseudo)
    random real number R in (0,1) and set S[j+1, i] = u S[j, i] with u > rho + 1
    if R < q and S[j+1, i] = d S[j, i] with d < 1 if R > q.

    ...

    Attributes
    ----------
    initialValue : float
        the initial value S(0) of the process
    decreaseIfDown : float
        the number d such that S(j+1)=S(j) with probability 1 - q
    increaseIfUp : float
        the number u such that S(j+1)=S(j) with probability q
    numberOfTimes : int
        the number of times at which the process is simulated, initial time
        included
    interest rate : double 
        the interest rate rho such that the risk free asset B follows the dynamics
        B(j+1) = (1+rho)B(j)
    riskNeutralProbabilityUp : float
        the risk neutral probability q =(1+rho-d)/(u-d) such that
        P(S(j+1)=S(j)*u) = q, P(S(j+1)=S(j)*d) = 1 - q,
        u > rho+1, d<1
    realizations : [double, double]
        a matrix containing the realizations of the process    
    numberOfSimulations : int
        the number of simulated trajectories of the process
    seed : int
        the seed to give to generate the sequence of (pseudo) random numbers
        which we use to generate the realizations of the process
        
        

    Methods
    -------
    generateRealizations()
        It generates and returns the matrix representing the realizations of the process.
    getRealizationsAtGivenTime(timeIndex)
        It returns the realizations of the process at time timeIndex
    getRealizations()
        It returns the realizations of the process.
    getDiscountedAverageAtGivenTime(timeIndex)   
        It returns the average of the process at time timeIndex discouted at time 0
    getEvolutionDiscountedAverage()
        It returns the evolution of the average of the process discounted at time 0
    printEvolutionDiscountedAverage()
        It prints the evolution of the average of the process discounted at time 0
    plotEvolutionDiscountedAverage()
        It plots the evolution of the average of the process discounted at time 0
    getProbabilityOfGainAtGivenTime(timeIndex)
        It returns the probability that (1+rho)^(-j)S(j)>S(0)  
    getEvolutionProbabilityOfGain()
        It returns the evolution of probability that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1 
    printEvolutionProbabilitiesOfGain()
        It prints the evolution of probability that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1 
    plotEvolutionProbabilitiesOfGain()
        It plots the evolution of probability that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1 
    getUpsAndDowns()
        It returns a matrix whose single entry is u > rho + 1 with probability
        equal to self.riskNeutralProbabilityUp and d < 1 with probability
        equal to 1 - self.riskNeutralProbabilityUp
    getPath(timeIndex)
        It returns the entire path of the process for a given simulation index
    printPath(timeIndex)
        It prints the entire path of the process for a given simulation index
    plotPath(timeIndex)
        It plots the entire path of the process for a given simulation index
    maximumAtGivenTime(timeIndex)
        It returns the maximum realization of the process at time timeIndex
    getEvolutionMaximum()
        It returns the evolution of the maximum of the realizations of the
        process at given times
    printEvolutionMaximum()
        It prints the evolution of the maximum of the realizations of the
        process at given times
    plotEvolutionMaximum()
        It prints the evolution of the maximum of the realizations of the
        process at given times
    
    """   
    
    def __init__(self, initialValue, decreaseIfDown, increaseIfUp,
                 numberOfTimes, numberOfSimulations,
                 interestRate = 0, #it is =0 if not specified
                 mySeed = 1897 #it is =1897 if not specified)
                 ):
        """
        Attributes
        ----------
        initialValue : float
            the initial value S(0) of the process
        decreaseIfDown : float
            the number d such that S(j+1)=S(j) with probability 1 - q
        increaseIfUp : float
            the number u such that S(j+1)=S(j) with probability q
        numberOfTimes : int
            the number of times at which the process is simulated, initial time
            included
        interest rate : double 
            the interest rate rho such that the risk free asset B follows the dynamics
            B(j+1) = (1+rho)B(j)  
        numberOfSimulations : int
            the number of simulated trajectories of the process
        seed : int
            the seed to give to generate the sequence of (pseudo) random numbers
            which we use to generate the realizations of the process
         """
        self.mySeed = mySeed
        self.numberOfSimulations = numberOfSimulations
        seed(mySeed)
        super().__init__(initialValue, decreaseIfDown, increaseIfUp,
                 numberOfTimes,interestRate)
    

    def getUpsAndDowns(self):
        """ 
        This method returns a matrix which will be used to generate the matrix
        representing the values of the process. 
        
        In particular, 
        S[i+1,j] = a[i,j]S[i,j] where a is the matrix returned by this method.
        
        Returns
        -------
        upsAndDowns : list
            a matrix whose single entry is u > rho + 1 if a (pseudo)
            random number R in (0,1) is smaller than q and d<1 if R>q.
        """
        
        u = self.increaseIfUp
        d = self.decreaseIfDown
        
        q = self.riskNeutralProbabilityUp
        
        #look at how we define the size of the matrix that we will return
        upsAndDowns = []
        
        #look at the syntax of range: we want timeIndex to start from 0 and being
        #SMALLER than numberOfTimes - 1
        for timeIndex in range(self.numberOfTimes):
            randomNumbers = np.random.uniform(0,1,self.numberOfSimulations)
            upsAndDownsAtTime = [u if x < q else d for x in randomNumbers]
            upsAndDowns.append(upsAndDownsAtTime)
        return upsAndDowns


    def generateRealizations(self):
        """
        It generates and returns the realizations of the process.
        
        The realizations are hosted in a matrix S whose row i represent the value
        of the process at time i for all the state of the world, and whose column
        j represents the path of the process for the simulation (or state of the world)
        j.  

        Returns
        -------
        array
            The matrix hosting the realizations of the process.
        """
        #maybe, in this case, dealing with arrays instead of lists. If realizations
        #are hosted in an array, it's easier to perform operations with them:
        #for example, remember that when you sum or multiply two lists, the
        #operation is not executed component-wise.
        realizations = np.full((self.numberOfTimes,self.numberOfSimulations),math.nan)     
        # first the initial values. Look at how we can fill a vector with a single
        # value in Python.
        realizations[0, :] = np.full((self.numberOfSimulations), self.initialValue)
        upsAndDowns = self.getUpsAndDowns()
        for timeIndex in range(1,self.numberOfTimes):
            for simulationIndex in range(self.numberOfSimulations):
                #S[i+1,j] = upsAndDowns[i,j]S[i,j]
                realizations[timeIndex, simulationIndex] = \
                    realizations[timeIndex - 1, simulationIndex]  * \
                        upsAndDowns[timeIndex - 1][simulationIndex]
        return realizations            
  
    
    def getRealizationsAtGivenTime(self, timeIndex):
        """
        It returns the realizations of the process at time timeIndex

        Parameters
        ----------
        timeIndex : int
            the time at which we want the realizations of the process for
            all the simulations

        Returns
        -------
        realizations : array
            a vector representing the realizations of the process at time timeIndex.

        """
        realizations = self.realizations;
        return realizations[timeIndex,:]


    def getPath(self, simulationIndex):
        """It returns the entire path of the process for a given simulation index

        Parameters
        ----------
        simulationIndex : int
           the simulation for which we want the path of the process

        Returns
        -------
        array
            a vector representing the evolution of the process for the given simulation

        """
        realizations = self.realizations;
        return realizations[:,simulationIndex]
  
    
    def printPath(self, simulationIndex):
        """
        It prints the entire path of the process for a given simulation index

        Parameters
        ----------
        simulationIndex : int
           the simulation for which we want the path of the process

        Returns
        -------
        None.

        """
        path = self.getPath(simulationIndex);

        print("The path for the", simulationIndex, "-th simulation is the following:")
        print()
        print('\n'.join('{:.3}'.format(realization) for realization in path))
        print()
        
        
    def plotPath(self, simulationIndex, numberOfPaths):
        """
        It plots the entire path of the process for a given simulation index

        Parameters
        ----------
        simulationIndex : int
           the simulation for which we want the path of the process

        Returns
        -------
        None.

        """
        for k in range(numberOfPaths):
            
            path = self.getPath(simulationIndex + k);
            plt.plot(path)
        plt.xlabel('Time')
        plt.ylabel('Realizations of the process')
        plt.show()           
        
        
    def getDiscountedAverageAtGivenTime(self, timeIndex):
        """
        Parameters
        ----------
        timeIndex : int
            The time at which we want the average of the realizations of the process,
            discounted at time 0.

        Returns
        -------
        float
            the average of the realizations of the process at time timeIndex,
            discounted at time 0.

        """
        realizationsAtTimeIndex = self.getRealizationsAtGivenTime(timeIndex);
        #look at the use of np.mean: we get the average of the elements of a list
        return (1 + self.interestRate) ** (-timeIndex) * np.mean(realizationsAtTimeIndex)

        
    def getProbabilityOfGainAtGivenTime(self, timeIndex):
        """
        Parameters
        ----------
        timeIndex : int
            The time j at which we want the probability that (1+rho)^(-j)S(j)>S(0) 
            with rho = self.interestRate

        Returns
        -------
        float
            the probability that (1+rho)^(-timeIndex)S(timeIndex)>S(0)
            with rho = self.interestRate

        """
        realizationsAtTimeIndex = self.getRealizationsAtGivenTime(timeIndex)
        #another use of this peculiar Python syntax: here we return a list whose j-th element
        #is 1 if S(0)*(1+rho)^timeIndex <= S[timeIndex,j] and 0 otherwise.
        #This is done by converting booleans into numbers.
        zeroAndOnes = [int(self.initialValue*((1+self.interestRate)**timeIndex) <= x)
                              for x in realizationsAtTimeIndex]
        #we the return the percentage
        return 100 * np.mean(zeroAndOnes)


    def getMaximumAtGivenTime(self, timeIndex):
        """
        It returns the maximum of the realizations of the process at time timeIndex

        Parameters
        ----------
        timeIndex : the time at which we want the maximum of the realizations
        of the process
        
        Returns
        -------
        float
            the maximum of the realizations of the process at time timeIndex.

        """
        realizationsAtTimeIndex = self.getRealizationsAtGivenTime(timeIndex)
        return np.max(realizationsAtTimeIndex)
 
        
    def getEvolutionMax(self):
        """
        It returns the evolution of the maximum of the realizations of the
        process at given times

        Returns
        -------
        evolutionMaximum : list
            a list representing the evolution of the maximum of the realizations of the
            process at given times.
        """
        evolutionMaximum = np.empty((self.numberOfTimes - 1))

        for timeIndex in range(0, self.numberOfTimes - 1):
            evolutionMaximum[timeIndex] = self.getMaximumAtGivenTime(timeIndex)
        
        return evolutionMaximum

    
    def printEvolutionMaximum(self):
        """
        It prints the evolution of the maximum of the realizations of the
        process at given times

        Returns
        -------
        None.

        """
        evolutionMaximum = self.getEvolutionMax();

        print("The path of the maximum evolution is the following:/n")
        print()
        print('\n'.join('{:.3}'.format(max) for max in evolutionMaximum))
        print()
    
        
    def plotEvolutionMaximum(self):
        """
        It plots the evolution of the maximum of the realizations of the
        process at given times

        Returns
        -------
        None.

        """
        evolutionMaximum = self.getEvolutionMax();
        plt.plot(evolutionMaximum)
        plt.xlabel('Time')
        plt.ylabel('Maximum realizations')
        plt.show()  
           
        
    
        