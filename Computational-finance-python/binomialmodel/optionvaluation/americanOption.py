#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Andrea Mazzon
"""
import numpy as np 

class AmericanOption:
    """
    This class is designed to valuate the price of an American option written on
    a binomial model for a given, general payoff. 
    
    It is also possible to get matrices representing the values of the option,
    the amount of money one would get by waiting and by exercising the
    option, respectively, and the exercise region. 
    
    Attributes
    ----------
    underlyingModel : binomial.creationandcalibration.BinomialModel
        the underlying binomialmodel   
        

    Methods
    -------
    getValueOption(self, payoffFunction, maturity)
        It returns the value of the american option of maturity maturity for
        the given payoff function 
   
    analysisOption(payoffFunction, maturity)
        For the given maturity and payoff function, it returns:
        - a matrix with the values of the american option at every time
        - a matrix with the amount of money one would get if he/she exercises
        - a matrix with the amount of money one would get if waiting
        - a matrix with 1 when it's convenient to exercise the option and 0 if
        it's convenient to wait
    """
    def __init__(self, underlyingProcess):
        self.underlyingProcess = underlyingProcess
        
    
    def getValueOption(self, payoffFunction, maturity):
        """
        It returns the value of the american option of maturity maturity for
        the given payoff function 

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff.
        maturity : int
            the maturity of the option.

        Returns
        -------
        float
            the value of the option.

        """
        
        binomialModel = self.underlyingProcess 
        
        q = binomialModel.riskNeutralProbabilityUp
        r = binomialModel.interestRate
        
        #we proceed backwards: we start from the payoff 
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        payoffRealizations = [payoffFunction(x) for x in processRealizations]
        
        #note that since here we are only interested to the price, we don't 
        #define any matrix but simply store the successive values of the option
        #in a vector that will be updated at every iteration of the for loop
        
        #at the beginning, the value of the option is equal to the payoff
        valuesOption = payoffRealizations
        
        for timeIndexBackward in range(maturity - 1,-1, -1):

            processRealizations = binomialModel.getRealizationsAtGivenTime(timeIndexBackward)               
            #the money we get if we exercise the option
            optionPart = [payoffFunction(x) for x in processRealizations]   
            
            #the money we get if we wait: 
            #V(j,k)=qV(j+1,k+1)+(1-q)V(j+1,k+1), where j is time and k the number
            #of ups up to time
            valuationPart = [(q * x + (1 - q) * y)/(1+r) for x,y in 
                              zip(valuesOption[0:(timeIndexBackward + 1)], 
                                  valuesOption[1:(timeIndexBackward + 2)])]    
            
            #and then we take the maximums: these are the current values of the option
            valuesOption = [max(x,y) for x,y in zip(optionPart, valuationPart)]    
        
        return valuesOption[0]
    
        
    
    
    def analysisOption(self, payoffFunction, maturity):
        """
        It performs an analysis of the american option option, returning 
        matrices representing the discounted values of the option, the discounted
        amount of money one would get by waiting and by exercising the option,
        respectively, and the exercise region.  

        Parameters
        ----------
        payoffFunction : lambda function 
            the function representing the payoff.
        maturity : int
            the maturity of the option.

        Returns
        -------
        valuesOption : list
            a triangular matrix with the discounted values of the american
            option at every time
        valuesExercise : list
            a triangular matrix with the discounted amount of money one would
            get if he/she exercises the option
        valuesIfWait : list
            a triangular matrix with the discounted amount of money one would
            get if waiting
        exercise : list
            a triangular matrix with 1 when it's convenient to exercise the
            option and 0 if it's convenient to wait.

        """
        
        binomialModel = self.underlyingProcess 
        q = binomialModel.riskNeutralProbabilityUp
        r = binomialModel.interestRate
        
        #here we store everything in some matrices, since we want to return
        #all the values over time. Of course, if we only want the price
        #we should call the method above
        valuesExercise = np.full((maturity + 1,maturity + 1),np.nan) 
        valuesIfWait = np.full((maturity + 1,maturity + 1),np.nan) 
        valuesOption = np.full((maturity + 1,maturity + 1),np.nan) 
        exercise = np.full((maturity + 1,maturity + 1),np.nan) 
        
        #we proceed backwards. We start from looking at the payoffs
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        payoffRealizations = [payoffFunction(x) for x in processRealizations]
        
        #all the values at maturity times are equal to the payoff
        valuesOption[maturity,:] = payoffRealizations
        valuesIfWait[maturity,:] = payoffRealizations
        valuesExercise[maturity,:] = payoffRealizations
        #and of course we exercise the option
        exercise[maturity,:] = np.full(maturity + 1, True)
        
        for timeIndexBackward in range(maturity - 1,-1, -1):

            processRealizations = binomialModel.getRealizationsAtGivenTime(timeIndexBackward)
            #the money we get if we exercise the option
            optionPart = [payoffFunction(x) for x in processRealizations]   
           
            #the money we get if we wait: 
            #V(j,k)=qV(j+1,k+1)+(1-q)V(j+1,k+1), where j is time and k the number
            #of ups up to time
            valuationPart = q/(1+r) * valuesOption[timeIndexBackward + 1, 0:(timeIndexBackward + 1)] + \
                (1-q)/(1+r) * valuesOption[timeIndexBackward + 1, 1:(timeIndexBackward + 2)]
                
     
            #and then we take the maximums: these are the current values of the option
            valuesOption[timeIndexBackward, 0:timeIndexBackward + 1] =\
                [max(x,y) for x,y in zip(optionPart, valuationPart)]      
            
            valuesExercise[timeIndexBackward, 0:timeIndexBackward + 1] = optionPart
            
            valuesIfWait[timeIndexBackward, 0:timeIndexBackward + 1] = valuationPart
            
            #this identifies the exercise region
            exercise[timeIndexBackward, 0:timeIndexBackward + 1] = \
                [x > y for x,y in zip(optionPart, valuationPart)]
                
        
        return valuesOption, valuesExercise, valuesIfWait, exercise
    
    