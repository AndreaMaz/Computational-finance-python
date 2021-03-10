#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import numpy as np

from pricingWithPDEs import PricingWithPDEs


class ExplicitEuler(PricingWithPDEs):
    """
    This class is devoted to numerically solve the PDE 
    
    U_t(x,t) - 1/2 x^2 sigma^2(x,t)U_{xx}(x,t) - r U_{x}(x,t) + r U(x,t) = 0,
        
    via Explicit Euler.
    
    This is the PDE corresponding to a local volatility model.
    
    Boundary conditions given as attributes of the class are applied at both ends
    of the domain. An initial condition is applied at t = 0. This can be seen
    as the payoff of an option. In this case, time represents maturity. 
        
    Attributes
    ----------
    dx : float
        discretization step on the space
    dt : float
        discretization step on the time
    xmin : float
        left end of the space domain
    xmax : float
        right end of the space domain
    tmax : float
        right end of the time domain
    x : float
        the dsicretized space domain
    numberOfSpaceSteps : int
        the number of intervals of the space domain
    numberOfTimeSteps : int
        the number of intervals of the time domain
    sigma : function
        the volatility function, depending on time and space
    r : float
        the interest rate
    payoff : function
        the initial condition. Called in this way because it corresponds to 
        payoff of an option seeing time as maturity
    functionLeft : function
        the condition at the left end of the space domain.
    functionRight : function
        the condition at the right end of the space domain.
    currentTime : int
        the current time. The PDE is solved going forward in time. Here the
        current time is used to plot the solution dynamically and to compute 
        the solution at the next time step in the derived classes.
                                                              
        

    Methods
    -------
    getSolutionAtNextTime():
        It returns the solution at the next time step
    solveAndPlot():
        It solves the PDE and dynamically plots the solution at every time step
        of length 0.1. It does not store the solution in a matrix
    solveAndSave():
        It solves the PDE and store the solution as a matrix in the self.solution
        attribute of the class. It also returns it.
    getSolutionForGivenMaturityAndValue(time, space):
        It returns the solution at given time and given space
    """
    
    
    def __init__(self, dx, dt, xmin, xmax, tmax, r, sigma, payoff, functionLeft, functionRight):
        """
        Parameters
        ----------
        dx : float
            discretization step on the space
        dt : float
            discretization step on the time
        xmin : float
            left end of the space domain
        xmax : float
            right end of the space domain
        tmax : float
            right end of the time domain
        sigma : function
            the volatility function, depending on time and space
        r : float
            the interest rate
        payoff : function
            the initial condition. Called in this way because it corresponds to 
            payoff of an option seeing time as maturity
        functionLeft : function
            the condition at the left end of the space domain
        functionRight : function
            the condition at the right end of the space domain
        currentTime : int
            the current time. The PDE is solved going forward in time. Here the
            current time is used to plot the solution dynamically and to compute 
            the solution at the next time step in the derived classes.

        Returns
        -------
        None.

        """
        self.sigma = sigma
        self.r = r
        super().__init__(dx, dt, xmin, xmax, tmax, payoff, functionLeft, functionRight)
        self.__initializeTerms()
        
       
    def __initializeTerms(self):
        #we initialize the multiplying terms for second and first space derivative
        self.multiplyTermFirstDerivative = 0.5 * self.dt/self.dx
        self.multiplySecondDerivative = self.dt/(self.dx*self.dx)
        
    def getSolutionAtNextTime(self):
        """    
        It returns the solution at the next time step, via Exlicit Euler
        Parameters
        ----------
        None

        Returns
        -------
        The solution at the next time step

        """
        uPast = self.uPast
        
        u = np.zeros((len(uPast)))
                       
        #this is zero for a call                                      
        u[0] = self.functionLeft(self.x[0], self.currentTime)
        
        #note that we have here central derivatives
        firstDerivatives = self.multiplyTermFirstDerivative * (uPast[2:]-uPast[:-2])
        
        secondDerivatives = self.multiplySecondDerivative * (uPast[2:]-2*uPast[1:-1]+uPast[:-2])
        
        u[1:-1] = uPast[1:-1] + 0.5 * secondDerivatives * self.sigma(self.x[1:-1])**2 \
            + firstDerivatives * self.r * self.x[1:-1] - self.dt * self.r * uPast[1:-1]
        
        #this is zero for a put 
        u[-1] = self.functionRight(self.x[-1], self.currentTime)
        return u
