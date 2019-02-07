import numpy as np
from sigmoid import *
from scipy.special import expit

#Hypothesis function and cost function for logistic regression
def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))

def lr_cost_function(mytheta, myX, myy, mylambda):
    m = myy.size

    # You need to return the following values correctly
    cost = 0
#    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    # =========================================================

    '''
    m = X.shape[0] #5000
    myh = h(theta,X) #shape: (5000,1)
    
    term1 = np.log( myh ).dot( -y ) #shape: (5000,5000)
    term2 = np.log( 1.0 - myh ).dot( 1 - y ) #shape: (5000,5000)
    left_hand = (term1 - term2) / m #shape: (5000,5000)
    right_hand = theta.dot( theta ) * lmd / (2*m) #shape: (1,1)
    return left_hand + right_hand #shape: (5000,5000)
    '''
    
    m = myX.shape[0] #5000
    myh = h(mytheta,myX) #shape: (5000,1)
    term1 = np.log( myh ).dot( -myy.T ) #shape: (5000,5000)
    term2 = np.log( 1.0 - myh ).dot( 1 - myy.T ) #shape: (5000,5000)
    left_hand = (term1 - term2) / m #shape: (5000,5000)
    right_hand = mytheta.T.dot( mytheta ) * mylambda / (2*m) #shape: (1,1)
    return left_hand + right_hand #shape: (5000,5000)

#    return cost, grad
