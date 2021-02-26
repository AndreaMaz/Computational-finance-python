#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import abc
import matplotlib.pyplot as plt

class BinomialModel(metaclass=abc.ABCMeta):
    """
    This is an abstract class whose main goal is to construct a binomial model.
    It can be extended by classes providing the way to simulate it

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
    realizations, [double, double]
        a matrix containing the realizations of the process     
        
        

    Methods
    -------
    generateRealizations()
        It generates the realizations of the process
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
    """
    #Python specific syntax for the constructor
    def __init__(self, initialValue, decreaseIfDown, increaseIfUp,
                 numberOfTimes,
                 interestRate = 0, #it is =0 if not specified
                 mySeed = 1897 #it is =1897 if not specified)
                 ):
        """
        Parameters
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
        interest rate : float 
            the interest rate rho such that the risk free asset B follows the dynamics
            B(j+1) = (1+rho)B(j)  
   
         """
        self.initialValue = initialValue
        self.decreaseIfDown = decreaseIfDown
        self.increaseIfUp = increaseIfUp
        self.interestRate = interestRate
        self.numberOfTimes = numberOfTimes
        self.riskNeutralProbabilityUp = (1 + interestRate - decreaseIfDown)/ (increaseIfUp - decreaseIfDown)
        #we generate the realizations once for all, during the call to the constructor
        self.realizations = self.generateRealizations()
        
    #note the syntax: this is an asbtract class, whose implementation will be
    #given in the derived classes. In our case, we will see the implementation
    #with a pure Monte Carlo method and a smarter one              
  
    @abc.abstractmethod
    def generateRealizations(self):
        """It generates the realizations of the process.
        """
        
        
    def getRealizations(self):
        """
        It returns the realizations of the process.

        Returns
        -------
        list
            The matrix hosting the realizations of the process.
        """ 
        return self.realizations   

    #this method is abstract as well: we will see indeed two different ways to
    #compute the average of the process, depending on the way the process is
    #generated
   
    @abc.abstractmethod
    def getDiscountedAverageAtGivenTime(self, timeIndex):
        """It computes the average of the realizations of the process at
        timeIndex, discounted at time 0
        
        Returns
        -------
        None.
        """
   
        
    def getEvolutionDiscountedAverage(self):
        """
        Returns
        -------
        list
            A vector representing the evolution of the average of the process
            discounted at time 0.   
        """
        
        #note this shortcup that Python offers us to run a for loop that 
        #generates a list
        return [self.getDiscountedAverageAtGivenTime(timeIndex) 
                for timeIndex in range(self.numberOfTimes)]
    
    
    def printEvolutionDiscountedAverage(self):
        """
        It prints the evolution of the average of the process discounted at time 0
        """
        evolutionDiscountedAverage = self.getEvolutionDiscountedAverage();

        print("The evolution of the average value of the discounted process is the following:")
        print()
        #note the syntax to tell the program we want to print three decimal digits,
        #separating the strings with a comma
        print('\n'.join('{:.3}'.format(discountedAverage) for discountedAverage
                        in evolutionDiscountedAverage))
        print()
    
    def plotEvolutionDiscountedAverage(self):
        """
        It plots the evolution of the average of the process discounted at time 0
        """
        evolutionDiscountedAverage = self.getEvolutionDiscountedAverage();
        
        plt.plot(evolutionDiscountedAverage)
        plt.xlabel('Time')
        plt.ylabel('Discounted average')
        plt.title('Evolution of the discounted average of the process')
        plt.show()  
       
       
    @abc.abstractmethod
    def getProbabilityOfGainAtGivenTime(self, timeIndex):
        """
        It returns the probability that (1+rho)^(-timeIndex)S(timeIndex)>S(0) 
        
        Parameters
        ----------
        timeIndex : int
            The time at which we want to get the probability
        Returns
        -------
        float
            The probability that (1+rho)^(-timeIndex)S(timeIndex)>S(0)
        """
        
    def getEvolutionProbabilityOfGain(self):
        """
        Returns
        -------
        list
            A list representing the evolution of probability that (1+rho)^(-j)S(j)>S(0), for
            j going from 1 to self.numberOfTimes - 1 .

        """
        
        return [self.getProbabilityOfGainAtGivenTime(timeIndex) 
                for timeIndex in range(self.numberOfTimes)]

    
    def printEvolutionProbabilitiesOfGain(self):
        """
        It prints the evolution of probability that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1 
        """
        probabilitiesPath = self.getEvolutionProbabilityOfGain();

        print("The path of the probability evolution  is the following:")
        print()
        print('\n'.join('{:.3}'.format(prob) for prob in probabilitiesPath))
        print()
 
    
    def plotEvolutionProbabilitiesOfGain(self):
        """
        "It plots the evolution of probability that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1 
        """
        probabilitiesPath = self.getEvolutionProbabilityOfGain();
        plt.plot(probabilitiesPath)
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title('Evolution of Q(S(j+1)>(1+rho)^jS(0)))')
        plt.show()     