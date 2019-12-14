import pandas as pd
import numpy as np
import itertools
import HigherOrderProducer
from sklearn.linear_model import LogisticRegression
import QDA

def performRegression(df, higherOrder=0):
    if higherOrder:
        df = HigherOrderProducer.createSquaredValues(df)
    logReg = LogisticRegression(multi_class='auto', solver='newton-cg', max_iter=1000)
    cols = df.columns
    data = QDA.partionData(df, 80)
    X = data[0].iloc[:, 0:len(cols)-1]
    Y = data[0].iloc[0:700, len(cols)-1:len(cols)].values
    XTest = data[1].iloc[:, 0:len(cols)-1]
    YTest = data[1].iloc[0:700, len(cols)-1:len(cols)].values
    Y = np.ravel(Y)
    model = logReg.fit(X, Y)
    print(model.score(XTest,YTest))

if __name__ == "__main__":
    frame = pd.DataFrame(np.array([[0,4,2],[0,2,2],[0,2,2], [0,6,2]]))
    cop = createSquaredValues(frame)
    print(cop)