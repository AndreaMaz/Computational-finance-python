#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import numpy as np
import scipy.stats as st

    
def blackScholesPriceCall(initialValue, r, sigma, T, strike):
    """
    It returns the analytical value of an european call option written
    on a Black-Scholes model.
   
    Parameters
    ----------
    numberOfSimulations : int
        the number of simulations of the the values of the process at maturity.
    initialValue : float
        the initial value of the process
    r : float
        the risk free rate
    sigma : float
        the log-volatility
    T : float
        the maturity of the option.
    strike : float
        the strike of the option.

    Returns
    -------
    callPrice : float
        the price of the call.
 
    """   

    d1 = (np.log(initialValue / strike) + (r + 0.5 * sigma ** 2) * T) / \
        (sigma * np.sqrt(T))
    d2 = (np.log(initialValue / strike) + (r - 0.5 * sigma ** 2) * T) / \
        (sigma * np.sqrt(T))
   
    callPrice = (initialValue * st.norm.cdf(d1, 0.0, 1.0) - \
        strike * np.exp(-r * T) * st.norm.cdf(d2, 0.0, 1.0))
   
    return callPrice

