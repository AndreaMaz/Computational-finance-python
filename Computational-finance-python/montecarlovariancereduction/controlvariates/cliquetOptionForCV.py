#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

from statistics import mean
from math import exp

class CliquetOptionForCV:
    """
    This class is designed to compute the payoff of a Cliquet option for a general
    matrix representing the returns of a process over sub-intervals.
    
    Its implmentation fits the application to control variates.
    
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
    
    def __init__(self, numberOfSimulations, maturity, localFloor, localCap):
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
        
        #the global floor and cap are not field of the class: we first get the
        #sum of the truncated returns, and we truncate it only when asked in
        #the method.
   
        #this is an attribute of the class: it gets initialized ONCE. Then, if
        #we consider the truncated option, we truncate its values
        self.nonTruncatedPayoffs = None
    
   
    def nonTruncatedPayoffSingleTrajectory(self, returns):
        """
        It returns the payoff of the Cliquet option with global floor = - infinity
        and global cap = infinity, for a specific simulation, not yet discounted

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
        nonTruncatedPayoff = sum(truncatedReturns)
                
        # we don't discount the payoff now. Can you guess why?
        return nonTruncatedPayoff
                
    #this is "private": with the double underscore as a prefix we make it possible
    #to call this method only by typing the name of the class
    def __setNonTruncatedPayoffs(self, returnsForAllSimulations):
                
        #we set the attribute of the class
        self.nonTruncatedPayoffs = [self.nonTruncatedPayoffSingleTrajectory(x) \
                                    for x in returnsForAllSimulations]
    
    def getPayoffs(self, returnsForAllSimulations, globalFloor = None, 
                   globalCap = None):
        """
        It returns the payoffs of the Cliquet option for all the simulations,
        not yet discounted

        Parameters
        ----------
        returnsForAllSimulations : list
            a matrix whose i-th row represents the returns for the i-th simulation
        global floor: float
            the global floor of the Cliquet option. Default None
        global cap: float
            the global cap of the Cliquet option. Default None       

        Returns
        -------
        payoff : float
            the payoffs of the Cliquet option for the all the simulations

        """
        #in this way, we initialize the attribute only once
        if self.nonTruncatedPayoffs is None:
            self.__setNonTruncatedPayoffs(returnsForAllSimulations)
                
        #in this case, we don't have global floor and cap. Ok, we are a bit lazy here..
        if globalFloor is None:
            payoffs = self.nonTruncatedPayoffs
        else: 
            payoffs = [min(max(x, globalFloor), globalCap) for x in self.nonTruncatedPayoffs]
        
        return payoffs
    
    
    def discountedPriceOfTheOption(self, returnsForAllSimulations, interestRate,
                                   globalFloor = None, globalCap = None):
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
        
        payoffs = self.getPayoffs(returnsForAllSimulations, globalFloor, globalCap)
         
        discountedPrice = exp(-interestRate * self.maturity) * mean(payoffs)
        
        return discountedPrice
         