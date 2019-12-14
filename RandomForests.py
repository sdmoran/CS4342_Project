import FeatureSelector as fs
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import DataOperations as do

if __name__ == "__main__":
    trainingData = pd.read_csv('./data/TrainData_Labeled.csv')

    # multDf = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Multiplicative.csv')
    multDf = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Squared.csv')  # Squared training data
    multTraining, multTesting = do.partionData(multDf, .8)
    rfc = RandomForestClassifier(n_estimators=500)
    bestFeatures = fs.getBestFeaturesForHigherOrderTerms(rfc, multTraining, 'best', 'accuracy')
    #bestFeatures = list(['alcohol', 'volatile acidity*total sulfur dioxide*density*', 'volatile acidity*chlorides*free sulfur dioxide*pH*', 'fixed acidity*volatile acidity*free sulfur dioxide*pH*sulphates*'])
    print(bestFeatures)

    trainingData = multTraining.loc[:, bestFeatures]
    trainingY = multTraining['label']
    trainingData.insert(loc = len(trainingData.columns),column='label', value=trainingY)

    testingData = multTesting.loc[:, bestFeatures]
    testingY = multTesting['label']
    testingData.insert(loc = len(testingData.columns),column='label', value=testingY)
    print(testingData)
    do.fitTrainingData(rfc, trainingData)
    do.testClassifier(rfc, testingData, "Random Forests")