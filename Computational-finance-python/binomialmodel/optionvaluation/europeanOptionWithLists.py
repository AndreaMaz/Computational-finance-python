#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import numpy as np 

class EuropeanOption:
    """
    The main goal of this class is to give the value of an European option with
    a general payoff at a given maturity. 
    
    It is also possible to get the values at any time before the maturity of
    an admissible portfolio, consisting of an amount of money in the risky asset
    and a one invested on the risk free asset, that must replicate the payoff at
    maturity. Moreover, the strategy (i.e., how much money is invested
    in the risky and risk free asset) can be got at any time before maturity.
    
    The values of the replicating portfolio are returned as a list.

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
        payoffFunction : function 
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which we want to valuate the discounted payoff.

        Returns
        -------
        discountedAverage : float
            the average of the realizations of payoffFunction(S(timeIndex))

        """
        discountedAverage = self.evaluatePayoff(payoffFunction, maturity)* \
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
        
        valuesPortfolio = []
        
        #realizations of the process at maturity
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        #payoffs at maturity
        payoffRealizations = [payoffFunction(x) for x in processRealizations]
        #the final values of the portfolio are simply the payoffs. Here we have 
        #to store the values from the last ones to the first ones. Then we 
        #revert the orders of the rows of valuesPortfolio.
        valuesPortfolio.append(payoffRealizations)

        #a small difference in the indexing with respect to before, due to the
        #fact that we simply append the realizations of the portfolio to the
        #list. Maybe a bit less intuitive.
        for timeBackFromMaturity in range(1, maturity + 1):    
            #V(k,j)=qV(k+1,j+1)+(1-q)V(k,j+1), with j current time, k number of
            #ups until current time
            valuesPortfolioAtTime = [q * x + (1-q) * y for (x,y) \
                in zip(valuesPortfolio[timeBackFromMaturity - 1][:-1], \
                       valuesPortfolio[timeBackFromMaturity - 1][1:])]
            valuesPortfolio.append(valuesPortfolioAtTime)
        
        # we revert the order of the rows of valuesPortfolio
        return valuesPortfolio[::-1]
      
        
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
        #here it's simpler with respect to before: we only have to specify the row,
        #the length is already set: the first row has length 1, the second one 2,
        # and so on.
        valuesPortfolioAtCurrentTime = allValuesPortfolio[currentTime]
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
        rho = binomialModel.interestRate
        
        valuesPortfolio = self.getValuesPortfolioBackwardAtGivenTime(payoffFunction, currentTime, maturity)
        #note that we cannot do discountedValuesPortfolioAtCurrentTime *  (1+rho)**(-(maturity - currentTime))
        #this operation is not allowed for lists (the multiplication by an int n)
        #is allowed, but in this case you just get a list which is made of n copies of your lists.
        discountedValuesPortfolioAtCurrentTime = [x * (1+rho)**(-(maturity - currentTime)) \
                                                       for x in valuesPortfolio]
 
        
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
        initialValuePortfolio = portfolioValues[0][0]
        
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
        rho = binomialModel.interestRate
               
        initialDiscountedValuePortfolio = self.getInitialValuePortfolio(payoffFunction, maturity)*(1+rho)**(-maturity)
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
        
        amountInRiskyAsset = []
        amountInRiskFreeAsset = []
        
        u = binomialModel.increaseIfUp
        d = binomialModel.decreaseIfDown
        rho = binomialModel.interestRate
        #same thing as in europeanOption, but with lists. Again, we have to
        #revert the order of the rows afterwards. 
        for timeBackFromMaturity in range(1, maturity + 1):
                        
            processAtNextTime = binomialModel.getRealizationsAtGivenTime(maturity - timeBackFromMaturity + 1)
            portfolioAtNextTime = \
                self.getValuesDiscountedPortfolioBackwardAtGivenTime(payoffFunction, maturity - timeBackFromMaturity + 1, maturity)
            
            currentAmountInRiskyAsset = \
            [(portfolioAtNextTime[k]-portfolioAtNextTime[k+1])/(processAtNextTime[k]-processAtNextTime[k+1])
                                  for k in range(maturity - timeBackFromMaturity + 1)]
            
            amountInRiskyAsset.append(currentAmountInRiskyAsset)
            
            currentAmountInRiskFreeAsset = [(u * portfolioAtNextTime[k+1] - d * portfolioAtNextTime[k])\
                                            /((u-d)*((1+rho)**(maturity - timeBackFromMaturity)))
                                  for k in range(maturity - timeBackFromMaturity + 1)]
            
            amountInRiskFreeAsset.append(currentAmountInRiskFreeAsset)
                
        
        return amountInRiskyAsset[::-1], amountInRiskFreeAsset[::-1]
    

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
        
        amountInRiskyAssetAtCurrentTime = amountInRiskyAsset[currentTime]
        amountInRiskFreeAssetAtCurrentTime = amountInRiskFreeAsset[currentTime]
       
        return amountInRiskyAssetAtCurrentTime, amountInRiskFreeAssetAtCurrentTime