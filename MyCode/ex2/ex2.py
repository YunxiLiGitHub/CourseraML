
import matplotlib
from costFunction import *
from plotDecisionBoundary import plot_decision_boundary 
import numpy as np
import matplotlib.pyplot as plt
import importlib as imp
from scipy.special import expit
from scipy import optimize

'''
def plot_data(X, y):
    plt.figure()

    plt.axis([30, 100, 30, 100])
    plt.legend(['Admitted', 'Not admitted'], loc=1)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    
    #Check to make sure I included all entries
    assert len(pos)+len(neg) == y.shape[0]
        
    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #
    plt.plot(X[pos,0],X[pos,1],'k+',label='Admitted')
    plt.plot(X[neg,0],X[neg,1],'bo',label='Admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)    
    plt.show()
'''

#Hypothesis function and cost function for logistic regression
def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))


#1.2.3
#An alternative to OCTAVE's 'fminunc' we'll use some scipy.optimize function, "fmin"
#Note "fmin" does not need to be told explicitly the derivative terms
#It only needs the cost function, and it minimizes with the "downhill simplex algorithm."
#http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.optimize.fmin.html
from scipy import optimize
def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin(computeCost, x0=mytheta, args=(myX, myy, mylambda), maxiter=400, full_output=True)
    return result[0], result[1]


def testEx2():
    cols = np.loadtxt('ex2data1.txt',delimiter=',',usecols=(0,1,2),unpack=False) #Read in comma separated data
    ##Form the usual "X" matrix and "y" vector
    X = cols[:,0:-1]
    y = cols[:,-1]
    m = y.shape[0]
#    plot_data(X,y)
    
    #1.2.1
    myx = np.arange(-10,10,.1)
    plt.plot(myx,expit(myx))
    plt.title("Woohoo this looks like a sigmoid function to me.")
    plt.grid(True)
#    plt.show()
  
    #1.2.2
    #Check that with theta as zeros, cost returns about 0.693:
    X = np.insert(X,0,1,axis=1)
    initial_theta = np.zeros((X.shape[1],1))
    print(computeCost(initial_theta,X,y))
    
    #1.2.3
    theta, mincost = optimizeTheta(initial_theta,X,y)
    #"Call your costFunction function using the optimal parameters of θ. 
    #You should see that the cost is about 0.203."
    print(computeCost(theta,X,y))
    
    #1.2.4
    plot_decision_boundary(X[:,1:], y, theta)
    plt.show()
    
    #For a student with an Exam 1 score of 45 and an Exam 2 score of 85, 
    #you should expect to see an admission probability of 0.776.
    print(h(theta,np.array([1, 45.,85.])))

    #Compute the percentage of samples I got correct:
    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    pos_correct = float(np.sum(makePrediction(theta,X[pos])))
    neg_correct = float(np.sum(np.invert(makePrediction(theta,X[neg]))))
    tot = len(pos)+len(neg)
    prcnt_correct = float(pos_correct+neg_correct)/tot
    print("Fraction of training samples correctly predicted: %f.", prcnt_correct)     
    
    #### 2.1 Visualizing the data
    return

if __name__ == "__main__":
    testEx2()
