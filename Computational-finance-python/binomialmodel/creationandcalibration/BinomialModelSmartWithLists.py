#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import numpy as np 
import scipy.special
import math
from binomialmodel.creationandcalibration.binomialModel import BinomialModel as BinomialModel


class BinomialModelSmartWithLists(BinomialModel):
    """
    In this class we implement the simulation of a binomial model by using 
    an approach that is computationally more effecient than Monte-Carlo, since
    it needs much less memory, and most of all is not prone to the accuracy problems
    that Monte Carlo exhibits.
    
    In particular, here we don't rely on pseudo random numbers generations, but
    at a given time we just consider ALL the possible realizations of the process
    attributing a (analytic!) probability to every realization. We then compute
    the average computing the weighted sum of the realizations with their probabilities.
    
    The realizations of the process are stored in an list representing a matrix.
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
    riskNeutralProbabilityUp : double
        the risk neutral probability q =(1+rho-d)/(u-d) such that
        P(S(j+1)=S(j)*u) = q, P(S(j+1)=S(j)*d) = 1 - q,
        u > rho+1, d<1
    realizations : [double, double]
        a matrix containing the realizations of the process    
             

    Methods
    -------
   
    getRealizations()
        It returns a list representing all the possible 
        realizations of the process up to time self.numberOfTimes - 1.   
    getRealizationsAtGivenTime(timeIndex)
        It returns the realizations of the process at time timeIndex
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
    getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
        It returns the probabilities corresponding to every possible realization
        of the process at time timeIndex.
    printProbabilitiesOfRealizationsAtGivenTime(timeIndex)
        It prints the probabilities corresponding to every possible realization
        of the process at time timeIndex.
   findThreshold(timeIndex):
        It returns the smallest integer k such that (u)^kd^(timeIndex-k) > 1,
        i.e., such that the realization of the process given by k ups and
        timeIndex - k downs, discounted at time 0, is bigger than the initial value.
    
   
    
    """   
    def __init__(self, initialValue, decreaseIfDown, increaseIfUp,
                 numberOfTimes,
                 interestRate = 0, #it is =0 if not specified
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
        super().__init__(initialValue, decreaseIfDown, increaseIfUp,
                 numberOfTimes,
                 interestRate)
        
    #this is supposed to be "private"
    def generateRealizations(self):
        #at every time N, there are N+1 possible values. The final time is
        #self.numberOfTimes - 1
        realizations = []  
        # the first entry of the list is a list itself, which only has one value.
        realizations.append([self.initialValue])
        for k in range(1, self.numberOfTimes):
            #the first realization is the previous first realization times u..
            highestRealization = self.increaseIfUp * realizations[k - 1][0]
            #the second is the previous first realization times d, and so on up
            #to the last one, which is the previous last one times d
            # here we cannot multiply directly by the vector, because it is a list
            realizationsAtTime = [self.decreaseIfDown * x for x in realizations[k - 1]]
            #the kth element of the list is a list representing the realizations
            #at time k: look at how we concatenate a list which is actually
            #a singleton with another list
            realizations.append([highestRealization] + realizationsAtTime)
        return realizations
    
    #this is supposed to be "public"
    def getRealizations(self):
        """
        It returns all the realizations of the process up to time self.numberOfTimes - 1.
        
        At every time N, there are N+1 possible reaalizations, depending on the
        number of times the process goes up: the "first" realization is given
        by N "ups", the second by N-1 ups and 1 down, etc. 
        
        The realizations are stored in a list which represents a triangular
        matrix 

        Returns
        -------
        realizations : list
            a triangular matrix storing all the possible realizations of the process up to
            time self.numberOfTimes - 1.
            This is a kind of particular matrix: the first row has length 1, the second 
            2, and so on.

        """
        
        return self.realizations
    
    
    def getRealizationsAtGivenTime(self, timeIndex):
        """
        It returns all the realizations of the process at time timeIndex

        Note that the first realization of the list is the one corresponding to
        all ups and no downs.
        
        Parameters
        ----------
        timeIndex : int
            the time at which we want the realizations of the process for
            all the simulations

        Returns
        -------
        list
            a vector representing the realizations of the process at time timeIndex.

        """
        
        #we return the element timeIndex of our list: this is the timeIndex-th
        #row of the trinagular matrix
        return self.realizations[timeIndex]
    
    
    def getProbabilitiesOfRealizationsAtGivenTime(self, timeIndex):
        """
        It returns the probabilities corresponding to every possible realization
        of the process at time timeIndex.
        
        Note that the first realization of the list is the one corresponding to
        all ups and no downs.
        
        The probabilities are computed using the fact that the realizations
        have Bernoulli distribution: in particular, the probability of the
        realization with k ups and N - k downs is
        N!/(k!(n-k)!)q^k(1-q)^(N-k), where q = self.riskNeutralProbabilityUp

        Parameters
        ----------
        timeIndex : int
            the time at which we want the probabilities.

        Returns
        -------
        list
            a vector representing the probabilities of every possible realization
            of the process at time timeIndex.

        """
        if timeIndex == 0:
            return 1.0
        else:
            q = self.riskNeutralProbabilityUp
            #NOTE: k represents the number of downs here
            probabilities = [scipy.special.binom(timeIndex, k) * q**(timeIndex - k) * (1-q)**k
                         for k in range(timeIndex + 1)]
            return probabilities
    
    
    def printProbabilitiesOfRealizationsAtGivenTime(self, timeIndex):
        """
        It prints the probabilities corresponding to every possible realization
        of the process at time timeIndex. 
        
        The probabilities are computed using the fact that the realizations
        have Bernoulli distribution: in particular, the probability of the
        realization with k ups and N - k downs is
        N!/(k!(n-k)!)q^k(1-q)^(N-k), where q = self.riskNeutralProbabilityUp

        Parameters
        ----------
        timeIndex : the time at which we want to print the probabilities

        Returns
        -------
        None.

        """
        probabilities = self.getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
        print("The probabilities of the realizations at time ", timeIndex, 
              " from the largest realizations to the smallest are ")        
        print('\n'.join('{:.3}'.format(prob) for prob in probabilities))
    
    
    def findThreshold(self, timeIndex):
        """
        It returns the smallest integer k such that (u)^kd^(timeIndex-k) > 1,
        i.e., such that the realization of the process given by k ups and
        timeIndex - k downs, discounted at time 0, is bigger than the initial value.

        Parameters
        ----------
        timeIndex : int
            the time at which we want to compute such a threshold.

        Returns
        -------
        int
            the smallest integer k such that (u)^kd^(timeIndex-k) > 1

        """
        rho = self.interestRate
        u = self.increaseIfUp
        d=self.decreaseIfDown
        return math.ceil(math.log(((1+rho)/d)**timeIndex,u/d))
    
    
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
        if timeIndex == 0:
            return 100.0
        else:
            probabilities = self.getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
            threshold = self.findThreshold(timeIndex)
            #we sum the probabilities corresponding to all the realizations with
            #enough ups k, and then we take the percentage
            #we have + 1 because 0:n is 0,1,..,n-1. We want to consider all the
            #realizations with anumber of downs <= timeIndex - threshold
            return 100.0 * sum(probabilities[0:timeIndex - threshold + 1])
    

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
        if timeIndex == 0:
            return self.initialValue
        else:
            probabilities = self.getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
            realizations = self.getRealizationsAtGivenTime(timeIndex)  
            #we discount the weighted sum of the realizations
            #use np.dot(a,b) to get the scalar product of two vectors arrays a, b
            discountedAverage = (1 + self.interestRate)**(-timeIndex) * np.dot(probabilities, realizations)
            return discountedAverage