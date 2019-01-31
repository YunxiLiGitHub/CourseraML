import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plotData import *
import costFunctionReg as cfr
import plotDecisionBoundary as pdb
import predict as predict
import mapFeature as mf
from scipy import optimize
from costFunction import *
from plotData import *
from mapFeature import *

#plt.ion()
plt.ioff()
# Load data
# The first two columns contain the exam scores and the third column contains the label.
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

#2.1 Visualizing the data
plot_data(X, y)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])
#plt.show()


#2.2 
# Add polynomial features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
mappedX = mf.map_feature(X[:, 0], X[:, 1])

#2.3
# Initialize fitting parameters
initial_theta = np.zeros(mappedX.shape[1])

# Set regularization parameter lambda to 1
lmd = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost, grad = cfr.cost_function_reg(initial_theta, mappedX, y, lmd)

#2.3.1
#I noticed that fmin wasn't converging (passing max # of iterations)
#so let's use minimize instead

def optimizeRegularizedTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.minimize(computeCost, mytheta, args=(myX, myy, mylambda),  method='BFGS', options={"maxiter":500, "disp":False} )
    return np.array([result.x]), result.fun
    
theta, mincost = optimizeRegularizedTheta(initial_theta,mappedX,y)


# 2.4 Plotting the decision boundary
def plotBoundary(mytheta, myX, myy, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta, mincost = optimizeRegularizedTheta(mytheta,myX,myy,mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            myfeaturesij = map_feature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta,myfeaturesij.T)
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")



#Build a figure showing contours for various values of regularization parameter, lambda
#It shows for lambda=0 we are overfitting, and for lambda=100 we are underfitting
plt.figure(figsize=(12,10))
plt.subplot(221)
plot_data(X, y)
plotBoundary(theta,mappedX,y,0.)

plt.subplot(222)
plot_data(X, y)
plotBoundary(theta,mappedX,y,4.)

plt.subplot(223)
plot_data(X, y)
plotBoundary(theta,mappedX,y,10.)

plt.subplot(224)
plot_data(X, y)
plotBoundary(theta,mappedX,y,100.)

plt.show()

# ===================== Part 2: Regularization and Accuracies =====================
# Optional Exercise:
# In this part, you will get to try different values of lambda and
# see how regularization affects the decision boundary
#
# Try the following values of lambda (0, 1, 10, 100).
#
# How does the decision boundary change when you vary lambda? How does
# the training set accuracy vary?
#

# Initializa fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lmd = 1

# Optimize
def cost_func(t):
    return cfr.cost_function_reg(t, X, y, lmd)[0]

def grad_func(t):
    return cfr.cost_function_reg(t, X, y, lmd)[1]

theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=initial_theta, maxiter=400, full_output=True, disp=False)

# Plot boundary
print('Plotting decision boundary ...')
pdb.plot_decision_boundary(theta, X, y)
plt.title('lambda = {}'.format(lmd))

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Compute accuracy on our training set
p = predict.predict(theta, X)

print('Train Accuracy: {:0.4f}'.format(np.mean(y == p) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

input('ex2_reg Finished. Press ENTER to exit')
