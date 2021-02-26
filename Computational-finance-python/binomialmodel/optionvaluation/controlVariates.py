#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import math
import numpy as np 
import scipy.stats as st

from binomialmodel.optionvaluation.americanOption import AmericanOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart

from europeanOption import EuropeanOption


class AmericanOptionWithControlVariates:
    """
    This class is devoted to an application of control variates to the
    approximation of the price of an american call and put option under the
    Black-Scholes model, simulating a binomial model. 
    
    In particular, we rely on: 
        - the analytic price P_E of an european call/put under the Black-Scholes
         model
        - the Monte-Carlo price P_E^N of an european call/put under the Black-Scholes
          model, simulating a binomial model with N times
        - the Monte-Carlo price P_A^N of an american call/put under the Black-Scholes
          model, simulating a binomial model with N times
          
    Using the heuristic that P_A - P_A^N is approximately equal to P_E - P_E^N,
    we then approximate P_A as P_A^N + P_E - P_E^N
    
    Attributes
    ----------
    initialValue : float
        the initial value of the process
    r : float
        the risk free rate of the model, i.e., such that the risk free asset B
        has dynamics B(t)=exp(r t) 
    sigma: float
        the log-volatility of the Black-Scholes model
    maturity: float
        the maturity of the option
    strike: float
        the strike of the option

    Methods
    -------
    blackScholesPriceCallAndPut():
        It returns the analytical value of an european call/put option written
        on a Black-Scholes model
   
    getAmericanCallAndPutPriceWithBinomialModel(numberOfTimes):
        It returns the approximated value of an european call/put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times
        
    getEuropeanCallAndPutPriceWithBinomialModel(numberOfTimes):
        It returns the approximated value of an european call/put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times
        
    getAmericanCallAndPutPriceWithControlVariates(self, numberOfTimes):
        It returns the approximated value of an american call/put option written
        on a Black-Scholes model, by using control variates
    """
    
    def __init__(self, initialValue, r, sigma, maturity, strike):
        """
         Attributes
         ----------
        initialValue : float
            the initial value of the process
        r : float
            the risk free rate of the model, i.e., such that the risk free asset B
            has dynamics B(t)=exp(r t) 
        sigma: float
            the log-volatility of the Black-Scholes model
        maturity: float
            the maturity of the option
        strike: float
            the strike of the option
        """
        
        self.initialValue = initialValue
        self.r = r
        self.sigma = sigma
        self.maturity = maturity
        self.strike = strike
    

    def blackScholesPriceCallAndPut(self):
        """
        It returns the analytical value of an european call/put option written
        on a Black-Scholes model

        Returns
        -------
        callPrice : float
            the price of the call.
        putPrice : float
            the proce of the put.

        """   
        initialValue = self.initialValue
        r = self.r
        sigma = self.sigma
        maturity = self.maturity
        strike = self.strike
        
        d1 = (np.log(initialValue / strike) + (r + 0.5 * sigma ** 2) * maturity) / \
            (sigma * np.sqrt(maturity))
        d2 = (np.log(initialValue / strike) + (r - 0.5 * sigma ** 2) * maturity) / \
            (sigma * np.sqrt(maturity))
    
        callPrice = (initialValue * st.norm.cdf(d1, 0.0, 1.0) - \
                     strike * np.exp(-r * maturity) * st.norm.cdf(d2, 0.0, 1.0))
        putPrice = (strike * np.exp(-r * maturity) * st.norm.cdf(-d2, 0.0, 1.0) -\
                    initialValue * st.norm.cdf(-d1, 0.0, 1.0))
        
        return callPrice, putPrice

    
    def getAmericanCallAndPutPriceWithBinomialModel(self, numberOfTimes):
        """
        It returns the approximated value of an american call/put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times

        Parameters
        ----------
        numberOfTimes : int
            the number of times where we simulate the binomial model.

        Returns
        -------
        priceCall : float
            the approximated price of the call.
        pricePut : float
            the approximated price of the put.

        """
        initialValue = self.initialValue
        r = self.r
        sigma = self.sigma
        maturity = self.maturity
        strike = self.strike
        
        increaseIfUp = math.exp(sigma * math.sqrt(maturity / numberOfTimes))
        decreaseIfDown = 1/increaseIfUp
 
        interestRate = math.exp(r * maturity / numberOfTimes) - 1
    
        binomialmodel = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 
    
        payoffEvaluator = AmericanOption(binomialmodel)
        
        payoffCall = lambda x : max(x-strike,0)
        payoffPut = lambda x : max(strike - x,0)
        
        priceCall = payoffEvaluator.getValueOption(payoffCall, numberOfTimes - 1)
        pricePut = payoffEvaluator.getValueOption(payoffPut, numberOfTimes - 1)
        
        return priceCall, pricePut


    def getEuropeanCallAndPutPriceWithBinomialModel(self, numberOfTimes):
        """
        It returns the approximated value of an european call/put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times

        Parameters
        ----------
        numberOfTimes : int
            the number of times where we simulate the binomial model.

        Returns
        -------
        priceCall : float
            the approximated price of the call.
        pricePut : float
            the approximated price of the put.

        """
        
        initialValue = self.initialValue
        r = self.r
        sigma = self.sigma
        maturity = self.maturity
        strike = self.strike
        
        increaseIfUp = math.exp(sigma * math.sqrt(maturity / numberOfTimes))
        decreaseIfDown = 1/increaseIfUp
 
        interestRate = math.exp(r * maturity / numberOfTimes) - 1
    
        binomialmodel = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 
    
        payoffEvaluator = EuropeanOption(binomialmodel)
        
        payoffCall = lambda x : max(x-strike,0)
        payoffPut = lambda x : max(strike - x,0)
        
        priceCall = payoffEvaluator.evaluateDiscountedPayoff(payoffCall, numberOfTimes - 1)
        pricePut = payoffEvaluator.evaluateDiscountedPayoff(payoffPut, numberOfTimes - 1)
        
        return priceCall, pricePut


    def getAmericanCallAndPutPriceWithControlVariates(self, numberOfTimes):
        """
        It returns the approximated value of an american call/put option written
        on a Black-Scholes model, by using control variates
        
        Parameters
        ----------
        numberOfTimes : int
            the number of times where we simulate the binomial model.

        Returns
        -------
        priceCall : float
            the approximated price of the call.
        pricePut : float
            the approximated price of the put.

        """
        
        bSCallEuropean, bSPutEuropean = self.blackScholesPriceCallAndPut()
        
        binomialCallEuropean, binomialPutEuropean = self.getEuropeanCallAndPutPriceWithBinomialModel(numberOfTimes)
        
        binomialCallAmerican, binomialPutAmerican = self.getAmericanCallAndPutPriceWithBinomialModel(numberOfTimes)
        
        controlVariateCallAmerican = binomialCallAmerican + (bSCallEuropean - binomialCallEuropean)
        controlVariatePutAmerican = binomialPutAmerican + (bSPutEuropean - binomialPutEuropean)
    
        return controlVariateCallAmerican, controlVariatePutAmerican