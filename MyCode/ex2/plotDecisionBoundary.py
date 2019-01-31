import matplotlib.pyplot as plt
import numpy as np
from plotData import *
from mapFeature import *

def plot_decision_boundary(X, y, theta):
    #Plotting the decision boundary: two points, draw a line between
    #Decision boundary occurs when h = 0, or when
    #theta0 + theta1*x1 + theta2*x2 = 0
    #y=mx+b is replaced by x2 = (-1/thetheta2)(theta0 + theta1*x1)

    boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
    boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)    
    plot_data(X, y)
    plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
    plt.legend()
    plt.show()    
    
