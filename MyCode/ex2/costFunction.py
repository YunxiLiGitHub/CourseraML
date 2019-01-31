import numpy as np
from predict import *

#Cost function, default lambda (regularization) 0
def computeCost(mytheta,myX,myy,mylambda = 0.): 
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    Note this includes regularization, if you set mylambda to nonzero
    For the first part of the homework, the default 0. is used for mylambda
    """
    m=myy.shape[0]
    #note to self: *.shape is (rows, columns)
    term1 = np.dot(-np.array(myy).T,np.log(predict(mytheta,myX)))
    term2 = np.dot((1-np.array(myy)).T,np.log(1-predict(mytheta,myX)))
    regterm = (mylambda/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:])) #Skip theta0
    return float( (1./m) * ( np.sum(term1 - term2) + regterm ) )
