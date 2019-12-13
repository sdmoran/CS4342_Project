import pandas as pd
import os
import DataOperations as do
import FeatureSelector as fs
from sklearn.linear_model import LogisticRegression
import AbstractedBestFeatureSelection as afs

numFeatures = 'best'
if __name__ == "__main__":
    lclf = LogisticRegression(multi_class='auto', solver='newton-cg', max_iter=1000)
    afs.doBestFeatureSelection(lclf, numFeatures)