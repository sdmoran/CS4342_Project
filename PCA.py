from sklearn.preprocessing import StandardScaler
import pandas as pd 
import os
from sklearn.decomposition import PCA
import DataOperations as do
import matplotlib.pyplot as plt

import sklearn.discriminant_analysis as da
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
featurePercentageThreshold = .9

def loadData():
    return pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Labeled.csv')

def findBestPCAFeatures(eigenValueRatios,threshold):
    sum = 0
    numFeatures = 0
    for value in eigenValueRatios:
        sum+=value
        numFeatures += 1
        if sum >= threshold:
            return numFeatures

def plotPcaComponentsAffectingY(x, y):
    for i in range(0, len(x.columns)):
        plt.title(f'How Principal Component {i} affects label')
        plt.plot(x.iloc[:,i], y, 'x')
        plt.show()
        print(i)

def getPCATraingAndTesting(thresh):
    allData = loadData()
    trainingData, testingData = do.partionData(allData, .8)
    trainingX = trainingData.loc[:, features]
    trainingY = trainingData.loc[:,'label']

    testingX = testingData.loc[:, features]
    testingY = testingData.loc[:, 'label']
    #Standardize features 
    #trainingX = StandardScaler().fit_transform(trainingX)

    pca = PCA()
    #Run PCA decomposition
    principalComponents = pca.fit_transform(trainingX)

    #Compute and print the number of components that PCA will extract
    numPcaComponents = findBestPCAFeatures(pca.explained_variance_ratio_, thresh)
    print(f'Components: {numPcaComponents}')

    principalDf = pd.DataFrame(principalComponents)
    trainingX = principalDf.iloc[:, 0:numPcaComponents+1]

    #Plot how each component affects the label
    #plotPcaComponentsAffectingY(principalDf, trainingY)

    testingX = pd.DataFrame(pca.transform(testingX))
    testingX = testingX.iloc[:, 0:numPcaComponents+1]

    return trainingX, trainingY, testingX, testingY


def testPCAOnDifferentClassifiers():
    qda = da.QuadraticDiscriminantAnalysis()
    trainingX, trainingY, testingX, testingY = getPCATraingAndTesting(featurePercentageThreshold)
    qda.fit(trainingX, trainingY)

    score = qda.score(testingX, testingY)
    print(f'QDA score: {score}')

    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(trainingX, trainingY)

    score = rfc.score(testingX, testingY)
    print(f'RandomForests: {score}')

    supportClf = svm.LinearSVC()
    supportClf.fit(trainingX, trainingY)
    score = supportClf.score(testingX, testingY)
    print(f'SVC Score: {score}')

    kNeighbor = KNeighborsClassifier()
    kNeighbor.fit(trainingX, trainingY)
    score = kNeighbor.score(testingX, testingY)
    print(f'KNearestNeighbors Score: {score}')

if __name__ == "__main__":
    testPCAOnDifferentClassifiers()