import FeatureSelector as fs
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import DataOperations as do

if __name__ == "__main__":
    multDf = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Multiplicative.csv')
    multTraining, multTesting = do.partionData(multDf, .8)
    rfc = RandomForestClassifier(n_estimators=100)
    bestFeatures = fs.getBestFeaturesForHigherOrderTerms(rfc, multTraining, 4, 'accuracy')
    print(bestFeatures)

    trainingData = trainingData.loc[:, bestFeatures]
    trainingY = multTraining['label']
    trainingData.insert(loc = len(trainingData.columns),column='label', value=trainingY)

    testingData = testingData.loc[:, bestFeatures]
    testingY = multTraining['label']
    testingData.insert(loc = len(testingData.columns),column='label', value=testingY)

    do.fitTrainingData(rfc, trainingData)
    do.testClassifier(rfc, testingData, "Random Forests")