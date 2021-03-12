#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import numpy as np
from finitedifferencemethods.pricingWithPDEs import PricingWithPDEs


class ImplicitEuler(PricingWithPDEs):
    """
    This class is devoted to numerically solve the PDE 
    
    U_t(x,t) - 1/2 x^2 sigma^2(x,t)U_{xx}(x,t) - r U_{x}(x,t) + r U(x,t) = 0,
        
    via Implicit Euler.
    
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
    
    def __init__(self, dx, dt, xmin, xmax, tmax, r, sigma, payoff,functionLeft, functionRight):
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
        
        #here we give the value of terms that will be useful to compute the scheme.
        
        self.multiplyTermFirstDerivative = 0.5 * self.dt/self.dx
        self.multiplySecondDerivative = self.dt/(self.dx*self.dx)
        
        #We initialize the matrix that defines the system to be solved.
        implicitEulerMatrix = np.zeros((self.numberOfSpaceSteps-1,self.numberOfSpaceSteps-1))
        #we use them to give the upper and lower diagonal
        i,j = np.indices(implicitEulerMatrix.shape)
        
        diagonal = 1 + self.multiplySecondDerivative * self.sigma(self.x[1:-1])**2 \
            + self.dt * self.r
        lowerDiagonal = - 0.5 * self.multiplySecondDerivative * self.sigma(self.x[2:-1])**2 \
            + self.r * self.multiplyTermFirstDerivative * self.x[2:-1]
        upperDiagonal = - 0.5 * self.multiplySecondDerivative * self.sigma(self.x[1:-2])**2 \
            - self.r * self.multiplyTermFirstDerivative * self.x[1:-2]

        #see how to define a multi-diagonal matrix
        implicitEulerMatrix[i==j] = diagonal
        implicitEulerMatrix[i==j+1] = lowerDiagonal
        implicitEulerMatrix[i==j-1] = upperDiagonal
   
        self.implicitEulerMatrix = implicitEulerMatrix
        
        
    def getSolutionAtNextTime(self):
        """    
        It returns the solution at the next time step, via Implicit Euler
        Parameters
        ----------
        None

        Returns
        -------
        The solution at the next time step

        """
        
        uPast = self.uPast # we are able to access the attribute of the parent class
        u = np.zeros((len(uPast)))
        
        #see the computations in the script
        knownTerm = uPast[1:-1]
        #uPast[0] is zero for a call option
        knownTerm[0] += (uPast[0]) \
           * (0.5 * self.multiplySecondDerivative * self.sigma(self.x[1])**2 \
              +  self.multiplyTermFirstDerivative * self.r * self.x[1])
               
        #uPast[-1] is zero for a put option
        knownTerm[-1] += (uPast[-1]) \
           * (0.5 * self.multiplySecondDerivative * self.sigma(self.x[-2])**2 \
              -  self.multiplyTermFirstDerivative * self.r * self.x[-2])
       
        #left boundary condition
        u[0] = self.functionLeft(self.x[0], self.currentTime)
        
        #solution by solving the system
        u[1:-1] = np.linalg.solve(self.implicitEulerMatrix, knownTerm)    
        #right boundary condition
        u[-1] = self.functionRight(self.x[-1], self.currentTime)
        
        return u
        
         
