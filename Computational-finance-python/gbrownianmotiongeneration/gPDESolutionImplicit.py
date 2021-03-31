#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import numpy as np

from finitedifferencemethods.pricingWithPDEs import PricingWithPDEs
from scipy.optimize import fsolve

class GPDESolutionImplicit(PricingWithPDEs):
    
    """
    This class is devoted to numerically solve via Implicit Euler the PDE 
    
    U_t(x,t) = G(U_xx(t,x)),
    U(0,x)=1_{x<=a},
    
    for fixed a in R, where G is the nonlinear G function
    
    G(y)=1/2 y sigma_u^2 if y >= 0, 
    G(y)=-1/2 y sigma_d^2 if y < 0, 
    
    for fixed 0<sigma_d<sigma_u.

    The solution of this PDE at (t,x)=(0,1) provides 
    
    E[1_{X<=a}],
    
    where E is the G-expectation and X is a G-normally distributed random
    variable.
            
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
    currentTime : int
        the current time. The PDE is solved going forward in time. Here the
        current time is used to plot the solution dynamically and to compute 
        the solution at the next time step in the derived classes.
                                                              
        

    Methods
    -------
    setThresholdForInitialCondition(self, threshold):
        It creates an object of the class with an initial condition 
        f(x) = 1_{x<=threshold} 
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
    
    
    def __init__(self, dx, dt, xmin, xmax, tmax, sigmaDown, sigmaUp):
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
        dtBrownianIncrement
        sigmaDown : float
            the left end of the volatility domain
        sigmaUp : float
            the right end of the volatility domain
        threshold : float
            the number a such that we want to compute E[1_{X<=a}].
        currentTime : int
            the current time. The PDE is solved going forward in time. Here the
            current time is used to plot the solution dynamically and to compute 
            the solution at the next time step in the derived classes.

        Returns
        -------
        None.

        """
        self.dx = dx # discretization step on the space
        self.dt = dt # discretization step on the time
        self.xmin = xmin
        self.xmax = xmax
        self.tmax = tmax
        
        self.gFunction = lambda x : 0.5 * sigmaUp**2 * x * (x>0) + \
            0.5 * sigmaDown**2 * x * (x <= 0)
        #self.gFunction = lambda x : 0.5 * sigmaUp**2 * x if x > 0 else 0.5 * sigmaDown**2 * x
        self.__initializeTerms()
        
       
    def __initializeTerms(self):
        #we initialize the multiplying terms for second and first space derivative
        self.multiplySecondDerivative = self.dt/(self.dx*self.dx)
        
        
    def setThresholdForInitialCondition(self, threshold):
        """
        It creates an object of the class with an initial condition 
        f(x) = 1_{x<=threshold} 

        Parameters
        ----------
        threshold : float 
            the number a such that the initial condition is f(x) = 1_{x<=a} .

        Returns
        -------
        None.

        """
        initialCondition = lambda x: (x <= threshold)
        super().__init__(self.dx, self.dt, self.xmin, self.xmax, self.tmax, initialCondition)    
        
        
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

        def equation(v):
            return v - np.concatenate(([self.multiplySecondDerivative * self.gFunction(v[1]-2 * v[0]+0)], \
                                [self.multiplySecondDerivative * self.gFunction(v[2:]-2*v[1:-1]+v[:-2])][:][0], \
                                    [self.multiplySecondDerivative * self.gFunction(1-2*v[-1]+v[-2])])) - uPast[1 : -1]                                 
        
        #condition at left border
        u[0] = 0
        
        u[1 : -1] = fsolve(equation, uPast[1 : -1])    
        
        #condition at right border                             
        u[-1] = 1

        return u