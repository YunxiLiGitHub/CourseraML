import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
#    plt.figure()

#    plt.axis([30, 100, 30, 100])
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
#    plt.legend()
    plt.grid(True)

