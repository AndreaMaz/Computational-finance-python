#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import numpy as np
import math 

class KnockOutOption:
    """
    The main goal of this class is to give the value of a Knock-out option with
    a general payoff at a given maturity. 
    
    ...

    Attributes
    ----------
    underlyingModel : binomial.creationandcalibration.BinomialModel
        the underlying binomialmodel   
        

    Methods
    -------
    
    getValuesPortfolioBackward(payoffFunction, maturity)
        It returns the values for every time until maturity of a portfolio
        replicating the payoff of the option at maturity.
    getValuesPortfolioBackwardAtGivenTime(payoffFunction, currentTime, finalTime):
        It returns the values at currentTime of a portfolio replicating the payoff
        of the option at maturity.
    getValuesDiscountedPortfolioBackwardAtGivenTime(payoffFunction, currentTime, finalTime):
        It returns the discounted values at currentTime of a portfolio replicating the payoff
        of the option at maturity.
    getInitialValuePortfolio(payoffFunction, maturity): 
        It returns the initial value of a portfolio replicating the payoff of
        the option at maturity.
    getInitialDiscountedValuePortfolio(payoffFunction, maturity): 
        It returns the initial value of a portfolio replicating the payoff of
        the option at maturity.
    getStrategy(self, payoffFunction, maturity):
        It returns two matrices, describing how much money must be invested in the
        risk free and in the risky asset at any time before maturity in order to
        replicate the payoff at maturity
     getStrategyAtGivenTime(self, payoffFunction, currentTime, maturity):
        It returns two vectors, describing how much money must be invested in the
        risk free and in the risky asset at currentTime in order to replicate
        the payoff at maturity
    """
    
    def __init__(self, underlyingModel):
        """
        
        Parameters
        ----------
        underlyingModel : binomial.creationandcalibration.BinomialModel
            the underlying binomial model 
        """
        self.underlyingModel = underlyingModel

    
    def getValuesPortfolioBackward(self, payoffFunction, maturity, 
                                   lowerBarrier = -np.inf, upperBarrier = np.inf):
        """
        It returns the values for every time until maturity of a portfolio
        replicating the payoff of the option at maturity.
        
        Note that the values are given as a triangular matrix, since at time
        k we have k +1 value of the portfolio. 

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the portfolio has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        valuesPortfolio : array
            a (triangular) matrix describing the value of the portfolio at
            every time before maturity

        """
        binomialModel = self.underlyingModel 
        q = binomialModel.riskNeutralProbabilityUp
        
        #we consider a number of times equal to maturity + 1
        valuesPortfolio = np.full((maturity + 1,maturity + 1),math.nan) 
        
        #realizations of the process at maturity
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        #payoffs at maturity
        payoffRealizations = [payoffFunction(x) if x > lowerBarrier and x < upperBarrier else 0
                              for x in processRealizations]
        #the final values of the portfolio are simply the payoffs
        valuesPortfolio[maturity,:] = payoffRealizations
        
        for timeIndexBackward in range(maturity - 1,-1, -1):  
            
            processRealizations = binomialModel.getRealizationsAtGivenTime(timeIndexBackward)
            #V(k,j)=qV(k+1,j+1)+(1-q)V(k,j+1), with j current time, k number of
            #ups until current time
            valuesPortfolio[timeIndexBackward,0: timeIndexBackward + 1] = \
            (q * valuesPortfolio[timeIndexBackward + 1, 0:timeIndexBackward + 1] +\
            (1-q) * valuesPortfolio[timeIndexBackward + 1, 1:timeIndexBackward + 2]) \
            * (processRealizations < upperBarrier).astype(int) * (processRealizations > lowerBarrier).astype(int)
        
        return valuesPortfolio
      
        
    def getValuesPortfolioBackwardAtGivenTime(self, payoffFunction, currentTime, maturity, 
                                   lowerBarrier = -np.inf, upperBarrier = np.inf):
        """
        It returns the values at currentTime of a portfolio replicating the payoff
        of the option at maturity.

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        currentTime : int
            the time at which we want the value of the portfolio
        maturity : int
            the time at which the portfolio has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        valuesPortfolioAtCurrentTime : array
            a vector describing the values of the portfolio at currentTime

        """
        allValuesPortfolio = self.getValuesPortfolioBackward(payoffFunction, maturity, lowerBarrier, upperBarrier)
        valuesPortfolioAtCurrentTime = allValuesPortfolio[currentTime, 0: currentTime + 1]
        return valuesPortfolioAtCurrentTime
    
            
    def getValuesDiscountedPortfolioBackwardAtGivenTime(self, payoffFunction, currentTime, maturity, 
                                   lowerBarrier = -np.inf, upperBarrier = np.inf):
        """
        It returns the discounted values at currentTime of a portfolio
        replicating the payoff of the option at maturity.

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        currentTime : int
            the time at which we want the discounted value of the portfolio
        maturity : int
            the time at which the portfolio has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        valuesPortfolioAtCurrentTime : array
            a vector describing the discounted values of the portfolio at currentTime

        """
        
        binomialModel = self.underlyingModel 
        rho = binomialModel.interestRate
        
        #Note that here we can multiply directly the vector by the float value.
        #This is not the case for lists
        discountedValuesPortfolioAtCurrentTime = \
            self.getValuesPortfolioBackwardAtGivenTime(payoffFunction, currentTime, maturity, lowerBarrier, upperBarrier) \
            *((1+rho)**(-(maturity - currentTime)))
        
        return discountedValuesPortfolioAtCurrentTime


    def getInitialValuePortfolio(self, payoffFunction, maturity, 
                                   lowerBarrier = -np.inf, upperBarrier = np.inf): 
        """
        It returns the initial value of a portfolio replicating the payoff of
        the option at maturity.
        
        This should be equal to the value returned by 
        evaluateDiscountedPayoff(payoffFunction, maturity):

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the portfolio has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        initialValuePortfolio : float
            the discounted value of the portfolio at initial time
        """
        
        portfolioValues = self.getValuesPortfolioBackward(payoffFunction, maturity, lowerBarrier, upperBarrier)
        initialValuePortfolio = portfolioValues[0,0]
        
        return initialValuePortfolio


    def getInitialDiscountedValuePortfolio(self, payoffFunction, maturity, 
                                   lowerBarrier = -np.inf, upperBarrier = np.inf):
        """
        It returns the discounted initial value of a portfolio replicating the payoff of
        the option at maturity.


        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the portfolio has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        initialDiscountedValuePortfolio : float
            the discounted value of the portfolio at initial time
        """
        
        binomialModel = self.underlyingModel 
        rho = binomialModel.interestRate
               
        initialDiscountedValuePortfolio = self.getInitialValuePortfolio(payoffFunction, maturity, lowerBarrier, upperBarrier)\
            *(1+rho)**(-maturity)
        return initialDiscountedValuePortfolio
    
    
   