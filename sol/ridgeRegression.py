import numpy as np
import regressData

##Question P3-1
import loadFittingDataP2
def ridgeExact(A, l, y):
    #Solves the regularized equation
    # Ax + lambda I= y
    ASquare = np.outer(A, A)
    print ASquare
    length = len(ASquare)
    identityMatrix = np.identity(length)
    #np.dot treats the vector as a column vector, as it is supposed to be.
    return np.dot(np.linalg.inv(l * identityMatrix +  ASquare) * np.transpose(A), y)

def question1(l):
    X,y = loadFittingDataP2.getData(False)
    return ridgeExact(X, l , y)
