import numpy as np
import matplotlib.pyplot as plt



def plotContours(X, t, lambdas, b, k, minRange, maxRange, step, filled):
    
    x,y = np.mgrid[minRange:maxRange:step,minRange:maxRange:step] 
    
    #been changed a good few times now 
    #https://mbhaskar1.github.io/machine%20learning/2019/07/04/svm-using-cvxopt.html 
    #https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning algorithms-in-python-3o1n3w07 page there i was trying to sort it with 
    #z = b + sum(cvx.mul(cvx.mul(lambdas,t),K(Xs,xvec)))
    
    if filled=="True":
        plt.contourf(x,y,z, levels=[-1e38, 0, 1e38], colors=["6DE1FF","FF568B"])
    plt.contour(x,y,z, levels=[-1, 0, 1], colors=["blue","green","red"], linewidths=[4])