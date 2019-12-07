import pandas as pd
import os 
import numpy as np
import sklearn.discriminant_analysis as da
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

def train_qda(allData):
    Y = np.array(allData['label'])
    X = np.array(allData.loc[:, allData.columns != 'label'])
    clf = da.QuadraticDiscriminantAnalysis()
    clf.fit(X,Y)
    return clf

def partionData(allData, percentage):
    trainingData = allData.loc[:percentage*len(allData)]
    testingData = allData.loc[percentage*len(allData):]
    return trainingData, testingData

def testQda(clf, testingData, label):
    Y = testingData['label']
    X = testingData.loc[:, testingData.columns != 'label']

    score = clf.score(X, Y)
    print("---------------------------------------")
    print(label)
    print(f'QDA Score: {score}')
    print("---------------------------------------")

def testBestFeatureQda(clf, testingLabels, testingFeatures):
    Y = np.array(testingLabels)
    X = np.array(testingFeatures)


def getBestFeaturesForQDA(trainingData):
    x = trainingData.iloc[:, 0:11]
    y = trainingData.iloc[:,11]
    bestFeatures = sfs( da.QuadraticDiscriminantAnalysis(),
        k_features="best",
        forward=False,
        floating=False,
        verbose=False,
        scoring='r2',
        ).fit(x,y)
    return bestFeatures.k_feature_names_, bestFeatures.k_feature_idx_

if __name__ == "__main__":
    allData = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Labeled.csv')
    trainingData, testingData = partionData(allData, .8)
    qdaClf = train_qda(trainingData)
    
    trainingY = trainingData['label']

    testQda(qdaClf, testingData, "No subset selection")

    #Feature selection
    bestFeatureNames, bestFeatureIdxs = getBestFeaturesForQDA(trainingData)

    bestFeaturesList = list(bestFeatureNames)
    bestFeaturesList.append('label')

    print(bestFeaturesList)

    bestFeaturesTrainingData = trainingData.loc[:, bestFeaturesList]

    bestFeaturesTestingData = testingData.loc[:, bestFeaturesList]

    print(f'Testing dataframe: \n{bestFeaturesTestingData}')

    bestFeaturesQda = train_qda(bestFeaturesTrainingData)

    testQda(bestFeaturesQda, bestFeaturesTestingData, "With forward subset selection")

