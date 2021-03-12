#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import abc
import numpy as np
import math
import matplotlib.pyplot as plt

class PricingWithPDEs(metaclass=abc.ABCMeta):
    """
    This class is devoted to numerically solve a general PDE 
    
        U_t + F[U_x,U_{xx},U–{xxx},..] = C(t),
        
    still not specifying the method, over the spatial domain of xmin <= x <= xmax
    that is discretized with a given step dx, and a time domain 0 <= t <= tmax
    that is discretized with a given time step dt.
    
    Boundary conditions given as attributes of the class are applied at both ends
    of the domain. An initial condition is applied at t = 0. This can be seen
    as the payoff of an option. In this case, time represents maturity. 
    
    It is extended by classes representing the specific PDE and the specific
    method.
    
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
    payoff : function
        the initial condition. Called in this way because it corresponds to 
        payoff of an option seeing time as maturity
    functionLeft : function
        the condition at the left end of the space domain. Default: None
    functionRight : function
        the condition at the right end of the space domain. Default:None
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
    
    def __init__(self, dx, dt, xmin, xmax, tmax, payoff, functionLeft = None, functionRight = None):
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
        payoff : function
            the initial condition. Called in this way because it corresponds to 
            payoff of an option seeing time as maturity
        functionLeft : function
            the condition at the left end of the space domain. Default: None
        functionRight : function
            the condition at the right end of the space domain. Default: None
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
        
        #we create an equi-spaced space discretization
        self.x = np.arange(self.xmin, self.xmax+self.dx, self.dx) 
        self.numberOfSpaceSteps = math.ceil((self.xmax-self.xmin)/self.dx)
        self.numberOfTimeSteps = math.ceil(self.tmax/self.dt)

        #conditions
        self.payoff = payoff
        self.functionLeft = functionLeft
        self.functionRight = functionRight
               
        self.__initializeU()
        
        self.currentTime = 0
        
        self.initialized = False
        
    def __initializeU(self):
        #here we initialize the solution, u0 stores the initial condition
        u0 = self.payoff(self.x)#x is an array: we can directly apply the payoff
        self.u = u0.copy()#if I say u = u0, u is THE SAME OBJECT as u0
        self.uPast = u0.copy()
    
    @abc.abstractmethod
    def getSolutionAtNextTime(self):
        """    
        It returns the solution at the next time step
        Parameters
        ----------
        None

        Returns
        -------
        The solution at the next time step

        """
        
    def solveAndPlot(self):
        """
        It solves the PDE and dynamically plots the solution at every time step
        of length 0.1. It does not store the solution in a matrix

        Returns
        -------
        None.

        """
        timeToPlot = 0 # we want to plot at times 0, 0.1, 0.2,..
        for i in range(self.numberOfTimeSteps+2):
            #we get the new solution. The solution will be computed in the
            #derived classes according to self.currentTime and self.uPast                           
            self.u = self.getSolutionAtNextTime()
            #and update uPast: u will be the "past solution" at the next time step
            self.uPast = self.u.copy()
            
            #we plot the solution when currentTime is (close to) 0.1, 0.2, ..
            if self.currentTime - self.dt < timeToPlot and self.currentTime >= timeToPlot:
                plt.plot(self.x, self.u, 'bo-', label="Numeric solution")
                #we assume here that the solution is not bigger than the max x
                #(generally true for options): then we set x[-1] to be the max
                #y axis
                plt.axis((self.xmin-0.12, self.xmax+0.12, 0, self.x[-1]))
                plt.grid(True)
                plt.xlabel("Underlying value")
                plt.ylabel("Price")
                plt.legend(fontsize=12)
                plt.suptitle("Time = %1.3f" % timeToPlot)
                plt.pause(0.01)
                timeToPlot += 0.1
            self.currentTime += self.dt

        plt.show()
        self.__initializeU()
        self.currentTime = 0


    def solveAndSave(self):
        """
        It solves the PDE and store the solution as a matrix in the self.solution
        attribute of the class. It also returns it.

        Returns
        -------
        array :
            The matrix representing the solution. Row k is the solution at time
            t_k = dt * k

        """
        solution = np.zeros((self.numberOfTimeSteps+1,self.numberOfSpaceSteps+1))
        for i in range(self.numberOfTimeSteps+1):
            #we store the solution at past time
            solution[i] = self.u           
            #we get the solution at current time. The solution will be computed in the
            #derived classes according to self.currentTime and self.uPast                    
            self.u = self.getSolutionAtNextTime()
            self.uPast = self.u.copy()  
            self.currentTime += self.dt
        self.solution = solution
        self.initialized = True #we want to compute all the solution only once.
        self.__initializeU()
        self.currentTime = 0
        return self.solution 
   
      
    def getSolutionForGivenTimeAndValue(self, time, space):
        """
        It returns the solution at given time and given space.

        Parameters
        ----------
        time : float
            the time: it represents maturiity for options
        space : float
            the space: it represents the underlying for options

        Returns
        -------
        float
            the solution at given time and space

        """
        #we generate the solution only once
        if not self.initialized:
           self.solveAndSave()
        solution = self.solution
        #we have to get the time and space indices
        timeIndexForTime = round(time/self.dt)
        spaceIndexForSpace = round((space - self.xmin)/self.dx)
        return solution[timeIndexForTime, spaceIndexForSpace]
        