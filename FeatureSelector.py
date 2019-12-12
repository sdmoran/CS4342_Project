from mlxtend.feature_selection import SequentialFeatureSelector as sfs

def getBestFeaturesForHigherOrderTerms(clf, trainingData, num_features, scoringString = 'r2'):
    x = trainingData.loc[:, trainingData.columns != 'label']
    y = trainingData.loc[:, 'label']
    bestFeatures = sfs( clf,
        k_features=num_features,
        forward=True,
        floating=False,
        verbose=2,
        n_jobs=8, # Use ALL the cores!
        scoring=scoringString,
        n_jobs=5
        ).fit(x,y)
    return bestFeatures.k_feature_names_

if __name__ == "__main__":
    pass