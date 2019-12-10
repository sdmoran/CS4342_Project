import numpy as np

def partionData(data, percentage):
    trainingData = data.loc[:percentage*len(data)]
    testingData = data.loc[percentage*len(data):]
    return trainingData, testingData

def fitTrainingData(clf, trainingData):
    Y = np.array(trainingData['label'])
    X = np.array(trainingData.loc[:, trainingData.columns != 'label'])
    clf.fit(X,Y)
    return clf

def testClassifier(clf, testingData, label):
    Y = testingData['label']
    X = testingData.loc[:, testingData.columns != 'label']

    score = clf.score(X, Y)
    print("---------------------------------------")
    print(label)
    print(f'Score: {score}')
    print("---------------------------------------")