#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
def bisection(f,leftEnd,rightEnd,tolerance,N):
    """
    It approximates the solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    leftEnd, rightEnd : floats
        The interval in which to search for a solution. The function returns
        None if f(leftEnd)*f(rightEnd) >= 0 since a solution is not guaranteed.
    tolerance : float
        the tolerance for the method: if f(midPoint) == tolerance for some
        midpoint midPoint = (currentLeftEnd + currentRightEnd)/2, then the function
        returns this solution
    N : integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval is [leftEnd,rightEnd]. If f(midPoint) == tolerance for some
        midpoint midPoint = (currentLeftEnd + currentRightEnd)/2, then the function
        returns this solution. If all signs of values f(leftEnd), f(rightEnd)
        and f(midPoint) are the same at any iteration, the bisection method fails
        and return None.
    """
    
    if f(leftEnd)*f(rightEnd) >= 0:
        print("Bisection method fails.")
        return None
    currentLeftEnd = leftEnd
    currentRightEnd = rightEnd
    for n in range(1,N+1):
        midPoint = (currentLeftEnd + currentRightEnd)/2
        f_midPoint = f(midPoint)
        if f(currentLeftEnd)*f_midPoint < 0:
            currentLeftEnd = currentLeftEnd
            currentRightEnd = midPoint
        elif f(currentRightEnd)*f_midPoint < 0:
            currentLeftEnd = midPoint
            currentRightEnd = currentRightEnd
        elif abs(f_midPoint) <= tolerance:
            return midPoint
        else:
            return None
    return (currentLeftEnd + currentRightEnd)/2