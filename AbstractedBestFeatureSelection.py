import pandas as pd
import os
import DataOperations as do
import FeatureSelector as fs

def doBestFeatureSelection(clf, numFeatures):
    multDf = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Labeled.csv')
    multTraining, multTesting = do.partionData(multDf, .8)
    bestFeatures = fs.getBestFeaturesForHigherOrderTerms(clf, multTraining, numFeatures, 'accuracy')
    #bestFeatures = list(['alcohol', 'volatile acidity*total sulfur dioxide*density*', 'volatile acidity*chlorides*free sulfur dioxide*pH*', 'fixed acidity*volatile acidity*free sulfur dioxide*pH*sulphates*'])
    print(bestFeatures)

    trainingData = multTraining.loc[:, bestFeatures]
    trainingY = multTraining['label']
    trainingData.insert(loc = len(trainingData.columns),column='label', value=trainingY)

    testingData = multTesting.loc[:, bestFeatures]
    testingY = multTesting['label']
    testingData.insert(loc = len(testingData.columns),column='label', value=testingY)
    print(testingData)
    do.fitTrainingData(clf, trainingData)
    do.testClassifier(clf, testingData, "Random Forests")