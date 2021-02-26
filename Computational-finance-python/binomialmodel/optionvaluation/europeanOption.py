#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import numpy as np 

class EuropeanOption:
    """
    This is an abstract class whose main goal is to give the value of an European
    option with a general payoff at a given maturity. 
    
    It is also possible to get the values at any time before the maturity of
    an admissible portfolio, consisting of an amount of money in the risky asset
    and a one invested on the risk free asset, that must replicate the payoff at
    maturity. Moreover, the strategy (i.e., how much money is invested
    in the risky and risk free asset) can be got at any time before maturity.

    ...

    Attributes
    ----------
    underlyingModel : binomial.creationandcalibration.BinomialModel
        the underlying binomialmodel   
        

    Methods
    -------
    evaluatePayoff(payoffFunction, maturity)
        It returns the average value of payoffFunction(S(maturity)), where
        S is the underlying binomial process
    evaluateDiscountedPayoff(payoffFunction, maturity)
        It returns the average value of (1+rho)^(-maturity) payoffFunction(S(maturity)),
        where S is the binomial process and rho is the interest rate, both
        got from the underlying model self.underlyingModel
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

        
    
    def evaluatePayoff(self, payoffFunction, maturity):
        """
        It returns the average value of payoffFunction(S(timeIndex)), where
        S is the underlying binomial process

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which we want to valuate the payoff.

        Returns
        -------
        average : float
            the average of the realizations of payoffFunction(S(maturity))

        """
        #just in order not to mention self.underlyingModel every time
        binomialModel = self.underlyingModel 
        
        #note one peculiar think of Python with respect for example to Java:
        #here you don't know the type of binomialModel, but still the compiler
        #does not complain if we call whatever method. This will be checked
        #instead at running time
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        payoffRealizations = [payoffFunction(x) for x in processRealizations]
        
        probabilities = binomialModel.getProbabilitiesOfRealizationsAtGivenTime(maturity)
        
        #as done in BinomialModelSmart, we compute the weighted sum of the
        #realizations with their probability
        average = np.dot(probabilities, payoffRealizations)
        
        return average
        
    
    def evaluateDiscountedPayoff(self, payoffFunction, maturity):
        """
        It returns the average value of payoffFunction(S(timeIndex)) discounted
        at initial time, where S is the underlying binomial process

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which we want to valuate the discounted payoff.

        Returns
        -------
        discountedAverage : float
            the average of the realizations of payoffFunction(S(timeIndex))

        """
        discountedAverage = self.evaluatePayoff(payoffFunction, maturity)*\
            (1+self.underlyingModel.interestRate)**(-maturity)
        return discountedAverage
    
    
    def getValuesPortfolioBackward(self, payoffFunction, maturity):
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

        Returns
        -------
        valuesPortfolio : list
            a (triangular) matrix describing the value of the portfolio at
            every time before maturity

        """
        binomialModel = self.underlyingModel 
        q = binomialModel.riskNeutralProbabilityUp
        
        #we consider a number of times equal to maturity + 1
        valuesPortfolio = np.empty((maturity + 1,maturity + 1)) 
        
        #realizations of the process at maturity
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        #payoffs at maturity
        payoffRealizations = [payoffFunction(x) for x in processRealizations]
        #the final values of the portfolio are simplythe payoffs
        valuesPortfolio[maturity,:] = payoffRealizations
        
        for timeIndexBackward in range(maturity - 1,-1, -1):    
            #V(k,j)=qV(k+1,j+1)+(1-q)V(k,j+1), with j current time, k number of
            #ups until current time
            valuesPortfolio[timeIndexBackward,0: timeIndexBackward + 1] = \
            q * valuesPortfolio[timeIndexBackward + 1, 0:timeIndexBackward + 1] + \
            (1-q) * valuesPortfolio[timeIndexBackward + 1, 1:timeIndexBackward + 2]
        
        return valuesPortfolio
      
        
    def getValuesPortfolioBackwardAtGivenTime(self, payoffFunction, currentTime, maturity):
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

        Returns
        -------
        valuesPortfolioAtCurrentTime : list
            a vector describing the values of the portfolio at currentTime

        """
        allValuesPortfolio = self.getValuesPortfolioBackward(payoffFunction, maturity)
        valuesPortfolioAtCurrentTime = allValuesPortfolio[currentTime, 0: currentTime + 1]
        return valuesPortfolioAtCurrentTime
    
            
    def getValuesDiscountedPortfolioBackwardAtGivenTime(self, payoffFunction, currentTime, maturity):
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

        Returns
        -------
        valuesPortfolioAtCurrentTime : list
            a vector describing the discounted values of the portfolio at currentTime

        """
        
        binomialModel = self.underlyingModel 
        r = binomialModel.interestRate
        
        discountedValuesPortfolioAtCurrentTime = \
            self.getValuesPortfolioBackwardAtGivenTime(payoffFunction, currentTime, maturity) \
            *((1+r)**(-(maturity - currentTime)))
        
        return discountedValuesPortfolioAtCurrentTime


    def getInitialValuePortfolio(self, payoffFunction, maturity): 
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

        Returns
        -------
        initialValuePortfolio : float
            the discounted value of the portfolio at initial time
        """
        
        portfolioValues = self.getValuesPortfolioBackward(payoffFunction, maturity)
        initialValuePortfolio = portfolioValues[0,0]
        
        return initialValuePortfolio


    def getInitialDiscountedValuePortfolio(self, payoffFunction, maturity):
        """
        It returns the discounted initial value of a portfolio replicating the payoff of
        the option at maturity.


        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the portfolio has to replicate the payoff

        Returns
        -------
        initialDiscountedValuePortfolio : float
            the discounted value of the portfolio at initial time
        """
        
        binomialModel = self.underlyingModel 
        r = binomialModel.interestRate
               
        initialDiscountedValuePortfolio = self.getInitialValuePortfolio(payoffFunction, maturity)*(1+r)**(-maturity)
        return initialDiscountedValuePortfolio
    
    
    def getStrategy(self, payoffFunction, maturity):
        """
        It returns two matrices, describing how much money must be invested in the
        risk free and in the risky asset at any time before maturity in order to
        replicate the payoff at maturity.
        
        Note that the values are given as a triangular matrix, since at time
        k we have k +1 values of the strategy. 
        
        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the portfolio has to replicate the payoff

        Returns
        -------
        amountInRiskyAsset : list
            a matrix describing how much money must be invested in the
            risky asset at any time before maturity in order to replicate the
            payoff at maturity.
        amountInRiskFreeAsset : list
            a matrix describing how much money must be invested in the
            risk free asset at any time before maturity in order to replicate the
            payoff at maturity.

        """
        binomialModel = self.underlyingModel
        
        amountInRiskyAsset = np.empty((maturity,maturity)) 
        amountInRiskFreeAsset = np.empty((maturity,maturity)) 
        
        u = binomialModel.increaseIfUp
        d = binomialModel.decreaseIfDown
        rho = binomialModel.interestRate
        
        for timeIndexBackward in range(maturity - 1,-1, -1):
                        
            processAtNextTime = binomialModel.getRealizationsAtGivenTime(timeIndexBackward + 1)
            portfolioAtNextTime = \
                self.getValuesDiscountedPortfolioBackwardAtGivenTime(payoffFunction, timeIndexBackward + 1, maturity)
            
            currentAmountInRiskyAsset = \
            [(portfolioAtNextTime[k]-portfolioAtNextTime[k+1])/(processAtNextTime[k]-processAtNextTime[k+1])
                                  for k in range(timeIndexBackward + 1)]
            
            amountInRiskyAsset[timeIndexBackward, 0 : timeIndexBackward + 1] = currentAmountInRiskyAsset 
            
            currentAmountInRiskFreeAsset = [(u * portfolioAtNextTime[k+1] - d * portfolioAtNextTime[k])\
                                            /((u-d)*((1+rho)**timeIndexBackward))
                                  for k in range(timeIndexBackward + 1)]
            
            amountInRiskFreeAsset[timeIndexBackward, 0 : timeIndexBackward + 1] = \
                currentAmountInRiskFreeAsset
        
        return amountInRiskyAsset, amountInRiskFreeAsset
    

    def getStrategyAtGivenTime(self, payoffFunction, currentTime, maturity):
        """
        It returns two vectors, describing how much money must be invested in the
        risk free and in the risky asset at currentTime in order to replicate
        the payoff at maturity
        
        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff we want to valuate.
        currentTime : int
            the time when we want to get the strategy
        maturity : int
            the time at which the portfolio has to replicate the payoff

        Returns
        -------
        amountInRiskyAssetAtCurrentTime : list
            a vector describing how much money must be invested in the
            risky asset at currentTime in order to replicate the payoff at
            maturity.
        amountInRiskFreeAssetAtCurrentTime : list
            a vector describing how much money must be invested in the
            risk free asset at currentTime in order to replicate the payoff at
            maturity.
        """
        
        amountInRiskyAsset, amountInRiskFreeAsset = self.getStrategy(payoffFunction, maturity)
        
        amountInRiskyAssetAtCurrentTime = amountInRiskyAsset[currentTime, 0 : currentTime + 1]
        amountInRiskFreeAssetAtCurrentTime = amountInRiskFreeAsset[currentTime, 0 : currentTime + 1]
       
        return amountInRiskyAssetAtCurrentTime, amountInRiskFreeAssetAtCurrentTime