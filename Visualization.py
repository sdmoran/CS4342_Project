# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from PIL import Image

data = pd.read_csv('./data/TrainData_Labeled.csv')

# Performs classification with given model and data.
# @param model: the SKLearn model to use for predictions
# @param data: The data to use for training and testing
def classify(model, data):
    # Convert data from Pandas DataFrame to np array for X and y
    X = np.asarray(data)[:, :11]
    y = np.asarray(data)[:, 11]

    # Partition data into train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, )
    
    # Classify with given model and print report
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # print(classification_report(y_test, pred))
    scores = cross_val_score(model, X_test, y_test, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))

def performPlots(data):
    X = np.asarray(data)[:, :11]
    y = np.asarray(data)[:, 11]
    dataToPlot = [] #array which holds arrays of each variable
    for i in range(0, 11):
        dataToPlot.append([])
    dataToClass = []
    for i in range(0, len(X)):
        for j in range(0, len(X[i])):
            dataToPlot[j].append(X[i][j])
        dataToClass.append(y[i])
        #plt.plot(datax, datay, 'x')
    newArray = groupData(dataToPlot, dataToClass)
    for arr in range(0, len(newArray)):
        datax = [item[0] for item in newArray[arr]]
        datay = [item[1] for item in newArray[arr]]
        color = colorFromIndex(arr+3)
        plt.plot(datax, datay, 'x', label=arr+3, color=color)
    #plt.plot(dataToPlot[i], dataToPlot)
    #plt.axis('equal')
    plt.legend()
    plt.savefig("pls.png")
    plt.show()
    #im = Image.open("pls.png") 
    #im.show()

def colorFromIndex(index):
    if index == 3:
        return '#ffcccc'
    elif index == 4:
        return '#99ff99'
    elif index == 5:
        return '#00ffff'
    elif index == 6:
        return '#ff5050'
    elif index == 7:
        return '#003366'
    else:
        return '#000000'

def groupData(dataToPlot, dataToClass):
    firstParam = dataToPlot[0]
    secondParam = dataToPlot[1]
    arr = []
    for i in range(0, 6):
        arr.append([])
    for j in range(0, len(firstParam)):
        arr[int(dataToClass[j])-3].append([firstParam[j], secondParam[j]])
    #print(arr[1])
    return arr
    
Heatmap correlations between different variables.
plt.figure(figsize=(12, 12))
sns.heatmap(data=data.corr(), annot=True)

# Classify with Stochastic Gradient Descent & print report
sgd = SGDClassifier(penalty=None)
print("Stochastic Gradient Descent classifier results:")
classify(sgd, data)

# Classify with K Nearest Neighbors & print report
knn = KNeighborsClassifier(n_neighbors = 5)
print("K Nearest Neighbors classifier results:")
classify(knn, data)

# Classify with Random Forest Classifier & print report
rfc = RandomForestClassifier(n_estimators=250)
print("Random Forest classifier results:")
classify(rfc, data)

performPlots(data)
