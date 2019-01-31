import numpy as np
from scipy.special import expit


def makePrediction(mytheta, myx):
    return predict(mytheta,myx) >= 0.5

def predict(theta, X):
    return expit(np.dot(X,theta))
    return p
