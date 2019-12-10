import pandas as pd
import os 
import numpy as np
import sklearn.discriminant_analysis as da
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import HigherOrderProducer as mop

def train_qda(allData):
    Y = np.array(allData['label'])
    X = np.array(allData.loc[:, allData.columns != 'label'])
    clf = da.QuadraticDiscriminantAnalysis()
    clf.fit(X,Y)
    return clf

def partionData(data, percentage):
    trainingData = data.loc[:percentage*len(data)]
    testingData = data.loc[percentage*len(data):]
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

def getBestFeaturesForHigherOrderTerms(trainingData, num_features):
    x = trainingData.loc[:, trainingData.columns != 'label']
    y = trainingData.loc[:, 'label']
    bestFeatures = sfs( da.QuadraticDiscriminantAnalysis(),
        k_features=num_features,
        forward=True,
        floating=False,
        verbose=2,
        scoring='r2',
        ).fit(x,y)
    return bestFeatures.k_feature_names_

if __name__ == "__main__":
    # allData = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Labeled.csv')
    # trainingData, testingData = partionData(allData, .8)
    # qdaClf = train_qda(trainingData)
    
    # trainingY = trainingData['label']

    # testQda(qdaClf, testingData, "No subset selection")

    # #Feature selection
    # bestFeatureNames, bestFeatureIdxs = getBestFeaturesForQDA(trainingData)

    # bestFeaturesList = list(bestFeatureNames)
    # bestFeaturesList.append('label')

    # print(bestFeaturesList)

    # bestFeaturesTrainingData = trainingData.loc[:, bestFeaturesList]

    # bestFeaturesTestingData = testingData.loc[:, bestFeaturesList]

    # print(f'Testing dataframe: \n{bestFeaturesTestingData}')

    # bestFeaturesQda = train_qda(bestFeaturesTrainingData)

    # testQda(bestFeaturesQda, bestFeaturesTestingData, "With forward subset selection")
    
    multDf = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Multiplicative.csv')
    multTraining, multTesting = partionData(multDf, .8)

    bestFeatures = getBestFeaturesForHigherOrderTerms(multTraining, 4)
    #bestFeatures = list(['volatile acidity*pH*', 'density*alcohol*', 'volatile acidity*citric acid*pH*', 'volatile acidity*density*sulphates*', 'free sulfur dioxide*pH*alcohol*', 'volatile acidity*total sulfur dioxide*density*sulphates*', 'citric acid*residual sugar*density*sulphates*alcohol*'])
    bestDfX = multTraining.loc[:,bestFeatures]
    trainingY = multTraining['label']
    bestDfX.insert(loc = len(bestDfX.columns),column='label', value=trainingY)
    bestFeaturesQda = train_qda(bestDfX)

    testingY = multTesting.loc[:,'label']
    bestDfTesting = multTesting.loc[:, bestFeatures]
    bestDfTesting.insert(loc = len(bestDfTesting.columns),column='label', value=testingY)

    testQda(bestFeaturesQda,bestDfTesting,f'Testing with labels {bestFeatures}')

    print(f'Test\n {bestDfTesting}\nTestY\n{trainingY}')
    