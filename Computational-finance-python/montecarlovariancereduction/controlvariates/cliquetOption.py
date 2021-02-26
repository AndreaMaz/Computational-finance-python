#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

from statistics import mean
from math import exp

class CliquetOption:
    """
    This class is designed to compute the payoff of a Cliquet option for a general
    matrix representing the returns of a process over sub-intervals.
    
    In particular, if X=(X)_{0 <= t <= T} is the underlying process, T is the
    maturity of the option and 0=t_0 <= t_1 <=..<= t_N =T is the partition of
    the interval, the payoff of the Cliquet option with local floor localFloor, 
    local cap localCap, global floor globalFloor and global cap globalCap is
    
    min(max(R_1^*+R_2^*+..+R_N^*, globalFloor), globalCap),
    
    where 
        
    R_k^* = min(max(R_k, localFloor), localCap)
    
    with
    
    R_k = X_{t_k}/X_{t_{k-1}} - 1.
    
    Attributes
    ----------
    numberOfSimulations : int
        the number of simulated trajectories of the returns
    maturity : float
            the maturity of the option
    localFloor : float
        the floor for the single return in the option
    localCap : float
        the cap for the single return in the option
    globalFloor : float
        the floor for the sum of the truncated returns
    globalcap : float
        the floor for the sum of the truncated returns 
        

    Methods
    -------
    payoffSingleTrajectory(returns):
        It returns the payoff of the Cliquet option for a specific simulation,
        not yet discounted
   
    discountedPriceOfTheOption(returnsForAllSimulations, interestRate):
        It returns the discounted price of the Cliquet option, as the discounted
        average of the payoffs for a single simulation of the returns.
        
    getPayoffs(self, returnsForAllSimulations):
        It returns the payoffs of the Cliquet option for all the simulations,
        not yet discounted
  
    """
    
    def __init__(self, numberOfSimulations, maturity, localFloor, localCap, globalFloor, globalCap):
        """
        Parameters
        ----------
        numberOfSimulations : int
            the number of simulated trajectories
        maturity : float
            the maturity of the option
        localFloor : float
            the floor for the single return in the option
        localCap : float
            the cap for the single return in the option
        globalFloor : float
            the floor for the sum of the trunctaed returns
        globalCap : float
            the floor for the sum of the trunctaed returns      

        Returns
        -------
        None.

        """
        self.numberOfSimulations = numberOfSimulations
        
        self.maturity = maturity
        
        self.localFloor = localFloor
        self.localCap = localCap
        
        self.globalFloor = globalFloor
        self.globalCap = globalCap
   
    
    def payoffSingleTrajectory(self, returns):
        """
        It returns the payoff of the Cliquet option for a specific simulation,
        not yet discounted

        Parameters
        ----------
        returns : list
            a vector representing the returns of the underlying process for one
            specific simulation

        Returns
        -------
        payoff : float
            the payoff of the Cliquet option for the specific simulation

        """
        
        truncatedReturns = [min(max(x - 1, self.localFloor), self.localCap) for x in returns]
        
        #you see how simply we can get the sum of elements of a list
        payoff = min(max(sum(truncatedReturns), self.globalFloor), self.globalCap)
        
        # we don't discount the payoff now. Can you guess why?
        return payoff
    
    
    def getPayoffs(self, returnsForAllSimulations):
        """
        It returns the payoffs of the Cliquet option for all the simulations,
        not yet discounted

        Parameters
        ----------
        returnsForAllSimulations : list
            a matrix whose i-th row represents the returns for the i-th simulation

        Returns
        -------
        payoff : float
            the payoffs of the Cliquet option for the all the simulations

        """
        
        #here x runs into the rows of the matrix returnsForAllSimulations..
        #in our mind! The compiler does not know that returnsForAllSimulations
        #is a matrix, and not even that payoffSingleTrajectory accepts a vector
        #as an argument.
        return [self.payoffSingleTrajectory(x) for x in returnsForAllSimulations]
    
    
    def discountedPriceOfTheOption(self, returnsForAllSimulations, interestRate):
        """
        It returns the discounted price of the Cliquet option, as the discounted
        average of the payoffs for a single simulation of the returns.
    
        Parameters
        ----------
        returnsForAllSimulations : list
            a matrix whose i-th row represents the returns for the i-th simulation
        interestRate : float
            the interest rate with resepct to which the option is price of the
            option is discounted.

        Returns
        -------
        discountedPrice : float
            the discounted price of the option
        """
        
        payoffs = self.getPayoffs(returnsForAllSimulations)
         
        discountedPrice = exp(-interestRate * self.maturity) * mean(payoffs)
        
        return discountedPrice
         
    
  