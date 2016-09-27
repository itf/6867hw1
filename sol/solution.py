import numpy as np
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
    matrix = [convertPointToVectorBasis(basisFunction, x, sizeBasis) for x in X]
    return np.matrix([convertPointToVectorBasis(basisFunction, x, sizeBasis) for x in X])

def linearBasis(x,m):
    return x**m

##question P1
######################################################################
import loadParametersP1
gaussMean, gaussCov, quadBowlA, quadBowlB = loadParametersP1.getData()

gradientPoints = []

def gradientDescent(objFunc, gradFunc, initGuess, stepSize, threshold):
    prevGuess = initGuess
    while True:
        grad = gradFunc(prevGuess)
        gradientPoints.append(linalg.norm(grad))
        nextGuess = prevGuess - stepSize * grad
        print (prevGuess)
        if linalg.norm(objFunc(nextGuess) - objFunc(prevGuess)) < threshold:
            return nextGuess, objFunc(nextGuess)
        prevGuess = nextGuess

def gradientApprox(objFunc, dstep, x):
    derivative  = (objFunc(x + dstep) - objFunc(x)) / (dstep)
    return derivative

def objFuncGaussian(x):
    return -1 * multivariate_normal.pdf(x, mean=gaussMean, cov=gaussCov)

def gradFuncGaussian(x):
    return -1 * objFuncGaussian(x) * np.dot(linalg.inv(gaussCov), (x - gaussMean))

def testGaussianDescent1():
    print gradientDescent(objFuncGaussian, gradFuncGaussian, [100, 100], 1e7, 1e-15)

def objFuncQuadBowl(x):
    return np.dot(np.dot(np.transpose(x), quadBowlA),x) / 2 - np.dot(np.transpose(x), quadBowlB);

def gradFuncQuadBowl(x):
    return np.dot(quadBowlA, x) - quadBowlB;

def testQuadradicBowlDescent1():
    print gradientDescent(objFuncQuadBowl, gradFuncQuadBowl, [0,0],1e-1, 1e-5)

def gaussianGradientApprox(x):
    dstep = 0.1;
    return gradientApprox(objFuncGaussian, 1, x)

def quadBowlradientApprox(x):
    dstep = 0.1;
    return gradientApprox(objFuncQuadBowl, 1000, x)

def testGaussianDescent2():
    print gradientDescent(gaussianGradientApprox, gradFuncGaussian, np.array([100, 100]), 1e7, 1e-15)

def testQuadradicBowlDescent2():
    print gradientDescent(objFuncQuadBowl, quadBowlradientApprox, np.array([0,0]),1e-2, 1e-5)

##testQuadradicBowlDescent2()
##plt.plot(gradientPoints)
##plt.ylabel('norm of gradient')
##plt.xlabel('step')
##plt.title('Negative Gaussian')
##plt.show()

##write your code here

#######################################################3
##question P2

######################################################
##Question P3-1
import loadFittingDataP2
def ridgeExact(A, l, y):
    #Solves the regularized equation
    # Ax + lambda I= y
    A = np.atleast_2d(A) #makes sure that it works with 1d vectors
    ASquare = np.dot(A.transpose(), A)
    length = max(ASquare.shape)
    identityMatrix = np.identity(length)
    #np.dot treats the vector as a column vector, as it is supposed to be.
    
    answer = np.dot(np.dot(np.linalg.inv(l * identityMatrix +  ASquare), np.transpose(A)), y)
    print A
    #turn into a vector
    answer = np.array(answer).flatten() 
    return answer

def question3_plot(x,y,basisFunction, xVal =None, yVal =None, sizesBasis = [2,4,8,11], regressionFunction = ridgeExact, lambdas = [0, 0.0001,0.01, 1]):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    if xVal == None:
        xVal = x
    if yVal == None:
        yVal = y
    xVal = np.array(xVal).flatten()
    yVal = np.array(yVal).flatten()

    positionsToPlot = np.linspace(min(xVal),max(xVal),100)
    plotN = 1
    plot = False
    plt.figure()
    for l in lambdas:
        plot = plt.subplot(len(lambdas)*100+10+plotN, sharex=plot if plotN>1 else None,  sharey=plot if plotN>1 else None)            
        if plotN < len(lambdas):
            plt.setp(plot.get_xticklabels() , visible=False)
        else:
            plt.xlabel("x")

        plt.plot(xVal,yVal, 'o', label = "Test Data")
        for sizeBasis in sizesBasis:
            xLinearBasis = convertVectorToMatrixBasis(basisFunction, x, sizeBasis)
            weights = regressionFunction(xLinearBasis, l , y)
            points = getPointsLinearFunction(weights, basisFunction, positionsToPlot)
            plt.title("lambda =" + str(l))
            plt.ylabel("f(x)")
            plt.plot(positionsToPlot,points, label = "Basis size = " +str(sizeBasis))
        if plotN ==1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=1)
        plotN = plotN + 1;
    plt.show()

def question3_error(x,y,basisFunction, xVal =None, yVal =None, sizesBasis = [2,4,8,11], regressionFunction = ridgeExact, lambdas = [0, 0.0001,0.01, 1]
):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    if xVal == None:
        xVal = x
    if yVal == None:
        yVal = y
    xVal = np.array(xVal).flatten()
    yVal = np.array(yVal).flatten()
    totalErrorSquare = {}
    plotN = 1
    plt.figure()
    for l in lambdas:
        plot = plt.subplot(len(lambdas)*100+10+plotN, sharex=plot if plotN>1 else None, sharey=plot if plotN>1 else None)
        if plotN < len(lambdas):
            plt.setp(plot.get_xticklabels() , visible=False)
        else:
            plt.xlabel("M")
        pointsErrors=[]
        for sizeBasis in sizesBasis:
            totalErrorSquare[(l,sizeBasis)]=0
            xLinearBasis = convertVectorToMatrixBasis(basisFunction, x, sizeBasis)
            weights = regressionFunction(xLinearBasis, l , y)
            points = getPointsLinearFunction(weights, basisFunction, xVal)
            totalErrorSquare[(l,sizeBasis)]+= sum([(yVal[i] - points[i])**2 for i in xrange(len(yVal))])
            pointsErrors.append(totalErrorSquare[(l,sizeBasis)])
        print pointsErrors
        plt.title("lambda =" + str(l))
        plt.ylabel("Error square")
        plt.plot(sizesBasis,pointsErrors)
        plotN = plotN + 1;
    plt.show()
    return totalErrorSquare

def question3_1():
    x,y = loadFittingDataP2.getData(False)
    basisFunction = linearBasis
    question3_plot(x,y, basisFunction)
    
#question3_1()
import regressData
def question3_2():
    xA,yA = regressData.regressAData()
    xB,yB = regressData.regressBData()
    xAll = np.append(xA,xB)
    yAll = np.append(yA,yB)
    xVal, yVal = regressData.validateData()
    lambdas = [0, 0.0001, 0.001, 0.01, 1]
    sizesBasis = [2,4,8,11]
    basisFunction = linearBasis
    error = question3_error(xA,yA,basisFunction, xAll, yAll, [2,3,4,5,6,7,8,9,10])
    question3_plot(xA,yA,basisFunction, xAll, yAll, [2,4,8,10])
    return error


#error = question3_2()
#[key for key,value in error.items() if value <= min(error.values()) +0.001]


## Start question 4
##################################################################################
## This question will use scikit learn magic
import lassoData
import math
from sklearn import linear_model

def sinBasis(x,m):
    if m == 0:
        return x
    else:
        return math.sin(0.4 * math.pi * x * m)

def lassoFunction(A, l, y):
    #returns the weights by using lasso as the regression function
    clf = linear_model.Lasso(alpha = l, fit_intercept=False)
    clf.fit(A,y)
    return clf.coef_

def lassoTrue():
    import pylab as pl
    data = pl.loadtxt('lasso_true_w.txt')
    return data.T

def question4PlotDataPretty():
    l = 0.01
    sizeBasis = 13
    xTrain, yTrain = lassoData.lassoTrainData()
    xVal, yVal = lassoData.lassoValData()
    xTest, yTest = lassoData.lassoTestData()
    basisFunction = sinBasis

    xTrain = np.array(xTrain).flatten()
    yTrain = np.array(yTrain).flatten()


    positionsToPlot = np.linspace(min(xVal-0.6),max(xVal+0.6),100)

    regressionFunction = lassoFunction
    
    
    xLinearBasis = convertVectorToMatrixBasis(basisFunction, xTrain, sizeBasis)    
    weights1 = regressionFunction(xLinearBasis, l , yTrain)
    points = getPointsLinearFunction(weights1, basisFunction, positionsToPlot)
    plt.plot(positionsToPlot,points, label = "LASSO ")


    regressionFunction = ridgeExact
    xLinearBasis = convertVectorToMatrixBasis(basisFunction, xTrain, sizeBasis)    
    weights2 = regressionFunction(xLinearBasis, l , yTrain)
    points = getPointsLinearFunction(weights2, basisFunction, positionsToPlot)
    plt.plot(positionsToPlot,points, label = "ridge ")



    realWeights = lassoTrue()

    points = getPointsLinearFunction(realWeights, basisFunction, positionsToPlot)
    plt.plot(positionsToPlot,points, label = "True ")
    
    plt.plot(xVal,yVal, 'ro', label = "valData ")
    plt.plot(xTest,yTest, 'kx', label = "testData ")
    plt.plot(xTrain,yTrain, 'g*', label = "trainData ")



    plt.legend(bbox_to_anchor=(1.05, 1), loc=1)
    plt.show()

    plt.figure()
    plt.bar(np.array(range(13))+0, weights1, 0.2, color ="blue", align = "center", label = "estimated W with LASSO ")
    plt.bar(np.array(range(13))+0.2, weights2, 0.2, color = "red", align = "center", label = "estimated W with ridge ")
    plt.bar(np.array(range(13)) +0.4, realWeights, 0.2, color = "black", align = "center",label = "W True ")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=1)
    plt.show()



    
def question4():
    xTrain, yTrain = lassoData.lassoTrainData()
    xVal, yVal = lassoData.lassoValData()
    xTest, yTest = lassoData.lassoTestData()

    ##Start processing the data
    sizesBasis = [13]
    lassoError = question3_error(xTrain, yTrain, sinBasis, xTest, yTest, sizesBasis, lassoFunction, [1e-8, 1e-4, 1e-2, 1e-1, 1])
    print lassoError[min(lassoError, key=lassoError.get)], min(lassoError, key=lassoError.get)
    question3_plot(xTrain, yTrain, sinBasis, xTest, yTest, sizesBasis, lassoFunction, [1e-8, 1e-4, 1e-2, 1e-1, 1])
    ridgeError = question3_error(xTrain, yTrain, sinBasis, xTest, yTest, sizesBasis, ridgeExact, [1e-8, 1e-4, 1e-2, 1e-1, 1])
    print ridgeError[min(ridgeError, key=lassoError.get)], min(ridgeError, key=lassoError.get)
    
question4PlotDataPretty()    
    
    
