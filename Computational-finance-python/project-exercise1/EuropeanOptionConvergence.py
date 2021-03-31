import math
import matplotlib.pyplot as plt
import numpy as np

from LogBinomialSmart import LogBinomialModelSmart
from binomialmodel.optionvaluation.europeanOption import EuropeanOption

from bachelierPDE import BachelierPDE

initialValue = 1
r = 0.0
sigma = 0.8

maturity = 5

#we are at the money
payoff = lambda x : np.maximum(x - initialValue, 0)

dx = 0.01
xmin = - 3
xmax = 3

dt = dx*dx/sigma**2
tmax = maturity

functionLeft = lambda x, t : 0
functionRight = lambda x, t : x - initialValue

solver = BachelierPDE(dx, dt, xmin, xmax, tmax, r, sigma, payoff, functionLeft, functionRight)

print("The price got solving the PDE is", solver.getSolutionForGivenTimeAndValue(maturity, initialValue))

#the maximum number of times N we use to approximate the price
maximumNumberOfTimes = 150

#we want to keep track of the prices and plot them
prices = []

#we also want to keep track of how u and d evolve for increasing N
increaseIfUps = []
decreaseIfDowns = []
payoffBin = lambda x : np.maximum(x - initialValue, 0)
for numberOfTimes in range (2, maximumNumberOfTimes + 1):
    
    increaseIfUp = math.exp(sigma * math.sqrt(maturity / numberOfTimes))
    decreaseIfDown = 1/increaseIfUp
    
    #we keep track of how u and d evolve for increasing N
    increaseIfUps.append(increaseIfUp)
    decreaseIfDowns.append(decreaseIfDown)
    
    interestRate = 0
    
    binomialmodel = LogBinomialModelSmart(math.exp(initialValue), decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 
    
    
    myPayoffEvaluator = EuropeanOption(binomialmodel)
    prices.append(myPayoffEvaluator.evaluatePayoff(payoffBin, numberOfTimes - 1))
    
  
plt.plot(prices)
plt.xlabel('Number of time steps')
plt.ylabel('Price')
plt.title("Price of a call option for a Bachelier model, approximated via log of a binomial model")
plt.ylim([prices[-1]*97/100, prices[-1]*102/100])
plt.show()      