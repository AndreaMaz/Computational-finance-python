#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import math
from scipy.optimize import fsolve
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart
from binomialmodel.optionvaluation.europeanOption import EuropeanOption


class BinomialModelCalibration:
    """
    This class performs a simple calibration of the binomial model, getting
    u > 1 + rho, where rho is the interest rate, and d < 1 such that
    S(j+1)=uS(j) with probability q and S(j+1)=dS(j) with probability d. 
    
    We get u and d exploiting the fact that q = (1 + rho - d)/ (u - d) 
    and that Var(log(S(j)/S(0)))=j^2 Var(log(xi)), where xi is the random variable
    that can be equal to u with probability q and d with probability 1 - q.   
    
    Attributes
    ----------
    initialValue : float
        the initial value S(0) of the process
    interestRate : float
        the interest rate rho such that the risk free asset B follows the dynamics
        B(j+1) = (1+rho)B(j)
    riskNeutralProbabilityUp : double
        the risk neutral probability q =(1+rho-d)/(u-d) such that
        P(S(j+1)=S(j)*u) = q, P(S(j+1)=S(j)*d) = 1 - q,
        u > rho + 1, d < 1   

    Methods
    -------
    calibrateFromLogVarianceForGivenTime(observedVariance, observationTime)
        It returns tha calibrated parameters u and d from the risk neutral probability
        uf ups and from the observed variance of S(observationTime)/S(0) at 
        time observationTime.
     testCalibration(decreaseIfDown, increaseIfUp, observationTime):
        It tests the goodness of the calibration first computing q = (1+rho-d)/(u-d)
        and Var(log(S(j)/S(0))) from two given u and d and then calibrating the
        values of u and d from q and Var(log(S(j)/S(0))). We can then see if 
        these are close to the original ones.
    """
    
    def __init__(self, interestRate, riskNeutralProbabilityToGoUp, initialValue):
        self.interestRate = interestRate
        self.riskNeutralProbabilityToGoUp = riskNeutralProbabilityToGoUp
        self.initialValue = initialValue
        """
        Parameters
        ----------
        initialValue : float
            the initial value S(0) of the process
        interestRate : float
            the interest rate rho such that the risk free asset B follows the dynamics
            B(j+1) = (1+rho)B(j)
        riskNeutralProbabilityUp : double
            the risk neutral probability q =(1+rho-d)/(u-d) such that
            P(S(j+1)=S(j)*u) = q, P(S(j+1)=S(j)*d) = 1 - q,
            u > rho + 1, d < 1
        """
  
        
    def calibrateFromLogVarianceForGivenTime(self, observedVariance, observationTime):
        """
        It returns tha calibrated parameters u and d from the risk neutral probability
        uf ups and from the observed variance of S(j)/S(0) at time j.
        
        Parameters
        ----------
        observedVariance : float
            the observed variance of the logarithm of the process divided by the
            initial value at the observation time.
        observationTime : int
            the time when the variance is observed.

        Returns
        -------
        2-uple
            the calibrated parameters u and d.

        """
        q = self.riskNeutralProbabilityToGoUp
        rho = self.interestRate
        #we define a function inside the method: we can do it!
        #we then give this function to the Python optimizer to find the zeros
        def equations(p):
            u, d = p
            #see the computations in the script
            return (1+rho-d)/(u-d) - q, math.log(u/d)**2 - observedVariance/(observationTime*q*(1-q)) 
        #note that we give the values from which we start to get u and d              
        return fsolve(equations, (1+rho+0.1 , 0.1))
                
    
    def testCalibration(self, decreaseIfDown, increaseIfUp, observationTime):
        """
        It tests the goodness of the calibration first computing 
        q = (1+rho-d)/(increaseIfUp-decreaseIfDown)
        and Var(log(S(j)/S(0))) from two given increaseIfUp and decreaseIfDown
        and then calibrating the values of increaseIfUp and decreaseIfDown from
        q and Var(log(S(j)/S(0))). We can then see if these are close to the
        original ones.

        Parameters
        ----------
        decreaseIfDown : float
            the number d such that S(j+1)=S(j) with probability 1 - q
        increaseIfUp : float
            the number u such that S(j+1)=S(j) with probability q
        observationTime : int
            the time when the variance is observed

        Returns
        -------
        2 - uple
            the calibrated u and d

        """
        interestRate = self.interestRate
        initialValue = self.initialValue
        #we construct the binomial model from the parameters..
        mybinomialmodelsmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                               observationTime + 1, interestRate) 
        
        #..and we want to compute the variance of log(S(observationTime)/S(0))
        varianceEvaluator = EuropeanOption(mybinomialmodelsmart)
      
        firstPayoff = lambda x : math.log(x/initialValue)**2
        secondPayoff = lambda x : math.log(x/initialValue)
        
        expectationFirstPayoff = varianceEvaluator.evaluatePayoff(firstPayoff, observationTime)
        expectationSecondPayoff = varianceEvaluator.evaluatePayoff(secondPayoff, observationTime)      
        
        #we get the variance..
        logVariance = expectationFirstPayoff - expectationSecondPayoff**2
       
        #..and we give it to calibrate the model
        u, d = self.calibrateFromLogVarianceForGivenTime(logVariance, observationTime)
        return u, d
