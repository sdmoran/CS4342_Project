import pandas as pd
import numpy as np
import itertools

def square(val):
    return val*val

def createSquaredValues(df):
    copy = df.copy()
    for col in copy.columns[0:len(df.columns)-1]:
        newName = str(col) + "Squared"
        newPosition = len(copy.columns) - 1
        copy.insert(newPosition, newName, list(map(square,copy.loc[:,col])))
    return copy
    
def createMultiplicativeTerms(df):
    pass

if __name__ == "__main__":
    frame = pd.DataFrame(np.array([[0,4,2],[0,2,2],[0,2,2], [0,6,2]]))
    cop = createSquaredValues(frame)
    print(cop)