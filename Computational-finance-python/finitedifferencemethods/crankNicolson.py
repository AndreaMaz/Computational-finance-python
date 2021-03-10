#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:25:03 2021

@author: Andrea Mazzon
"""

import numpy as np
from pricingWithPDEs import PricingWithPDEs


class CrankNicolson(PricingWithPDEs):
    """
    This class is devoted to numerically solve the PDE 
    
    U_t(x,t) - 1/2 x^2 sigma^2(x,t)U_{xx}(x,t) - r U_{x}(x,t) + r U(x,t) = 0,
        
    via Crank-Nicolson.
    
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
        self.numberOfSpaceSteps = round((self.xmax-self.xmin)/self.dx)
        self.initializeTerms()
        
       
    def initializeTerms(self):
        
        #here we give the value of terms that will be useful to compute the scheme.
        
        self.multiplyTermFirstDerivative = 0.5 * self.dt/self.dx
        self.multiplySecondDerivative = self.dt/(self.dx*self.dx)
        
        #We initialize the matrix that defines the system to be solved.
        implicitEulerMatrix = np.zeros((self.numberOfSpaceSteps-1,self.numberOfSpaceSteps-1))
        #we use them to give the upper and lower diagonal
        i,j = np.indices(implicitEulerMatrix.shape)
        
        #note that we respect to Implicit Euler we multiply the terms in the matrix
        #by 1/2, apart from the 1s. This is because the scheme is
        #(u_i^{n+1}-u_i^n)/dt = (F_i^{n+1}-F_i^n)/2
        diagonal = 1 + 0.5 * self.multiplySecondDerivative * self.sigma(self.x[1:-1])**2 \
            + 0.5 * self.dt * self.r
        lowerDiagonal = - 0.5 * self.multiplySecondDerivative * self.sigma(self.x[2:-1])**2 \
            + self.r * self.multiplyTermFirstDerivative * self.x[2:-1]
        upperDiagonal = - 0.5 * self.multiplySecondDerivative * self.sigma(self.x[1:-2])**2 \
            - self.r * self.multiplyTermFirstDerivative * self.x[1:-2]

        #see how to define a multi-diagonal matrix
        implicitEulerMatrix[i==j] = diagonal
        implicitEulerMatrix[i==j+1] = 0.5 * lowerDiagonal
        implicitEulerMatrix[i==j-1] = 0.5 * upperDiagonal
   
        self.implicitEulerMatrix = implicitEulerMatrix
        
        
    def getSolutionAtNextTime(self):
        """    
        It returns the solution at the next time step, via Crank-Nicolson
        Parameters
        ----------
        None

        Returns
        -------
        The solution at the next time step

        """
        uPast = self. uPast
        u = np.zeros((len(uPast)))
        
        firstDerivatives = self.multiplyTermFirstDerivative * (uPast[2:] - uPast[:-2])
        secondDerivatives = self.multiplySecondDerivative * (uPast[2:] - 2*uPast[1:-1] + uPast[:-2])
        
        
        addingTerm = 0.5 * secondDerivatives * self.sigma(self.x[1:-1])**2 \
                + firstDerivatives * self.r * self.x[1:-1] - self.dt * self.r * uPast[1:-1]
       
        #we multiply the adding term by 0.5 is because the scheme is given by
        #(u_i^{n+1}-u_i^n)/dt = (F_i^{n+1}-F_i^n)/2
        knownTerm = uPast[1:-1] + 0.5 * addingTerm
        
        #we multiply by 0.5 is because the scheme is given by
        #(u_i^{n+1}-u_i^n)/dt = (F_i^{n+1}-F_i^n)/2
        knownTerm[0] += 0.5 * uPast[0] \
           * (0.5 * self.multiplySecondDerivative * self.sigma(self.x[1])**2 \
              +  self.multiplyTermFirstDerivative * self.r * self.x[1])
           
        #we multiply by 0.5 is because the scheme is given by
        #(u_i^{n+1}-u_i^n)/dt = (F_i^{n+1}-F_i^n)/2
        knownTerm[-1] += 0.5 * uPast[-1] \
           * (0.5 * self.multiplySecondDerivative * self.sigma(self.x[-2])**2 \
              +  self.multiplyTermFirstDerivative * self.r * self.x[-2])
       
        u[0] = self.functionLeft(self.x[0], self.currentTime)
        
        u[1:-1] = np.linalg.solve(self.implicitEulerMatrix, knownTerm)    

        u[-1] = self.functionRight(self.x[-1], self.currentTime)
        
        return u
        
         
