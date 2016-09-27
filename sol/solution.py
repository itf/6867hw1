import numpy as np
import regressData
import matplotlib.pyplot as plt

##Utils

def plotPoints(points):
    XBasis = convertToBasis(basisFunction, X, sizeBasis)

def getPointsLinearFunction(weights, basisFunction, positions):
    sizeBasis = len(weights)
    points=[]
    fVectorX = lambda x : convertPointToVectorBasis(basisFunction, x, sizeBasis)
    evaluateX = lambda x: (np.dot(fVectorX(x), weights)) #converts to vector
    for x in positions:
        points.append(evaluateX(x))
    return points


##Used by question 2
def convertPointToVectorBasis(basisFunction, x, sizeBasis):
    return np.array([basisFunction(x,m) for m in xrange(sizeBasis)])

def convertVectorToMatrixBasis(basisFunction, X, sizeBasis):
    #this function returns a m by (len(X)) matrix
    
    #basisFunction is a function that takes 2 values:
    #x, m, and returns \phi_m(x) as defined in the book
    return np.matrix([convertPointToVectorBasis(basisFunction, x, sizeBasis) for x in X])

def linearBasis(x,m):
    return x**m

##question P1
##write your code here

##question P2


##Question P3-1
import loadFittingDataP2
def ridgeExact(A, l, y):
    #Solves the regularized equation
    # Ax + lambda I= y
    A = np.atleast_2d(A) #makes sure that it works with 1d vectors
    ASquare = np.dot(A.transpose(), A)
    length = len(ASquare)
    identityMatrix = np.identity(length)
    #np.dot treats the vector as a column vector, as it is supposed to be.
    answer = np.dot(np.linalg.inv(l * identityMatrix +  ASquare) * np.transpose(A), y)
    #turn into a vector
    answer = np.array(answer).flatten() 
    return answer

def question3_plot(x,y,basisFunction):
    positionsToPlot = np.linspace(min(x),max(x),100)
    plotN = 1
    lambdas = [0, 0.0001,0.01, 1]
    sizesBasis = [2,4,8,11]
    plot = False
    for l in lambdas:
        plot = plt.subplot(len(lambdas)*100+10+plotN, sharex=plot if plotN>1 else None)
        if plotN < len(lambdas):
            plt.setp(plot.get_xticklabels() , visible=False)
        else:
            plt.xlabel("x")

        plt.plot(x,y, 'o')
        for sizeBasis in sizesBasis:
            xLinearBasis = convertVectorToMatrixBasis(linearBasis, x, sizeBasis)
            weights = ridgeExact(xLinearBasis, l , y)
            points = getPointsLinearFunction(weights, basisFunction, positionsToPlot)
            plt.title("lambda =" + str(l))
            plt.ylabel("f(x)")
            plt.plot(positionsToPlot,points)

        plotN = plotN + 1;
    plt.show()
    
def question3_1():
    x,y = loadFittingDataP2.getData(False)
    basisFunction = linearBasis
    question3_plot(x,y, basisFunction)

def question3_1():
    x,y = loadFittingDataP2.getData(False)
    basisFunction = linearBasis
    question3_plot(x,y, basisFunction)
    
#question3_1()

def question3_2():
    
