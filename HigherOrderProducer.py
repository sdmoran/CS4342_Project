import pandas as pd
import numpy as np
import itertools

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
        combos = list(itertools.combinations(copyDf.columns[0:len(copyDf)-2], i))
        for combo in combos:
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
    frame = pd.DataFrame(np.array([[4,4,2],[0,2,2],[1,2,2], [3,6,2]]))
    cop = createSquaredValues(frame)
    cop = createMultiplicativeTerms(frame)
    print(cop)