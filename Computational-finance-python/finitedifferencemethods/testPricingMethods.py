#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we compare Explicit Euler, Implicit Euler and Crank-Nicholson on the valuation
of a call option. We compare them in terms of accuracy and time.

@author: Andrea Mazzon
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import time

from implicitEuler import ImplicitEuler
from explicitEuler import ExplicitEuler
from crankNicolson import CrankNicolson

from analyticformulas.analyticFormulas import blackScholesPriceCall


dx = 0.1
xmin = 0
xmax = 14

dt = dx 
tmax = 3

sigma = 0.5
sigmaFunction = lambda x : sigma*x
r = 0.4

strike = 2

payoff = lambda x : np.maximum(x - strike, 0)

functionLeft = lambda x, t : 0
#this is because from put-call parity and since the price of a put is close to
#zero for large values of the underyling, the price of the call for large values
# x of the underlying can be approximated by x - strike * math.exp(-r * T),
#where T is the maturity 
functionRight = lambda x, t : x - strike * math.exp(-r * t)

dtExplicitEuler = 0.01*dx*dx # the minimum value such that it is stable

explicitEulerSolver = ExplicitEuler(dx, dtExplicitEuler, xmin, xmax, tmax, r, sigmaFunction, payoff, functionLeft, functionRight)
implicitEulerSolver = ImplicitEuler(dx, dt, xmin, xmax, tmax, r, sigmaFunction, payoff, functionLeft, functionRight)
crankNicolsonSolver = CrankNicolson(dx, dt, xmin, xmax, tmax, r, sigmaFunction, payoff, functionLeft, functionRight)
    
def compareErrorsAndTimes():
    """
    It compares Explicit Euler, Implicit Euler and Crank-Nicholson on the valuation
    of a call option, in terms of accuracy and time. It prints the average error
    for different values of the underlying, from strike/2 and 2*strike, and the
    time needed to get the whole solution at the given time and space.

    Returns
    -------
    None.

    """
    
    errorExplicit = []
    errorImplicit = []
    errorCrankNicolson = []
    

    #we compute the times needed to compute the solutions
    timeExplicitInit = time.time() 
    explicitEulerSolver.solveAndSave()
    timeExplicit = time.time()  - timeExplicitInit
    timeImplicitInit = time.time() 
    implicitEulerSolver.solveAndSave()
    timeImplicit = time.time()  - timeImplicitInit
    timeCrankNicolsonInit = time.time() 
    crankNicolsonSolver.solveAndSave()
    timeCrankNicolson = time.time()  - timeCrankNicolsonInit

    print() 
    print("The elapsed time using Explicit Euler is ", timeExplicit)
    
    print() 
    print("The average elapsed time using Implicit Euler is ", timeImplicit)
    
    print() 
    print("The average elapsed time using Crank Nicolson is ", timeCrankNicolson)
    
    #and the errors
    for x in np.arange(strike/2, strike*2, 0.1) :
        analyticSolution = blackScholesPriceCall(x, r, sigma, tmax, strike)
        errorExplicit.append(abs(explicitEulerSolver.getSolutionForGivenMaturityAndValue(tmax, x)-analyticSolution) \
            / analyticSolution)
        errorImplicit.append(abs(implicitEulerSolver.getSolutionForGivenMaturityAndValue(tmax, x)-analyticSolution) \
            / analyticSolution)
        errorCrankNicolson.append(abs(crankNicolsonSolver.getSolutionForGivenMaturityAndValue(tmax, x)-analyticSolution) \
            / analyticSolution)

    print() 
    print("The average error using Explicit Euler is ", np.mean(errorExplicit))
    
    print() 
    print("The average error using Implicit Euler is ", np.mean(errorImplicit))
    
    print() 
    print("The average error using Crank Nicolson is ", np.mean(errorCrankNicolson))
    
   
   
    
def plotWithExactSolution():
    """
    It dynamically plots the solution got from the three methods, together with
    the analytic ones

    Returns
    -------
    None.

    """
    #we directly get the discretized space from one of the objects
    x = explicitEulerSolver.x
    
    for maturity in np.arange(0.1, tmax, 0.1):
        #we would have problems getting the Black-Scholes formula for x=0, since
        #it would divide by zero                                                     
        exactSolution = [0] + [blackScholesPriceCall(underlying, r, sigma, maturity, strike) for underlying in x[1:]] 
        
        solutionExplicitEuler = [explicitEulerSolver.getSolutionForGivenMaturityAndValue(maturity, underlying) for underlying in x]
        solutionImplicitEuler = [implicitEulerSolver.getSolutionForGivenMaturityAndValue(maturity, underlying) for underlying in x]
        solutionCrankNicolson = [crankNicolsonSolver.getSolutionForGivenMaturityAndValue(maturity, underlying) for underlying in x]

        plt.plot(x, exactSolution, 'r', label="Analytic solution")
        plt.plot(x, solutionExplicitEuler, 'bo-', label="Explicit Euler")
        plt.plot(x, solutionCrankNicolson, 'mo-', label="Crank-Nicolson")
        plt.plot(x, solutionImplicitEuler, 'go-', label="Implicit Euler")

        plt.axis((xmin-0.12, xmax+0.12, 0, x[-1]))
        plt.grid(True)
        plt.xlabel("Underlying value")
        plt.ylabel("Price")
        plt.legend(loc=1, fontsize=12)
        plt.suptitle("Maturity = %1.3f" % maturity)
        plt.pause(0.01)
            
        maturity += dt
    plt.show()
    
def main():
    #compareErrorsAndTimes()
    plotWithExactSolution()
    
if __name__ == "__main__":
    main()