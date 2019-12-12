import pandas as pd
import numpy as np
import itertools
import os 

def square(val):
    return val*val

def multiplicativeSum(numbers):
    sum = 1
    for number in numbers:
        sum = sum*number
    return sum

def createSquaredValues(df):
    copy = df.copy()
    for col in copy.columns[0:len(df.columns)-1]:
        newName = str(col) + "Squared"
        newPosition = len(copy.columns) - 1
        copy.insert(newPosition, newName, list(map(square,copy.loc[:,col])))
    return copy

def createMultiplicativeLabels(labels):
    name = ""
    for label in labels:
        name = name + str(label) + '*'
    return name

def createMultiplicativeTerms(df):
    copyDf = df.copy()
    for i in range(2, len(copyDf.columns)):
        combos = list(itertools.combinations(df.columns[0:len(df.columns)-1], i))
        for combo in combos:
            print(combo)
            comboList = list(combo)
            newName = createMultiplicativeLabels(comboList)
            newPosition = len(copyDf.columns) - 1
            comboDf = copyDf.loc[:, comboList]
            toInsert = list()
            for row in np.array(comboDf):
                toInsert.append(multiplicativeSum(row))
            copyDf.insert(newPosition, newName, toInsert)
            
    return copyDf

if __name__ == "__main__":
    allData = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/TrainData_Labeled.csv')
    #testingData = allData.iloc[1:10, :]
    testingData = createSquaredValues(allData)
    #Save multiplicative data to file
    testingData.to_csv('data/TrainData_Squared.csv', index = False)