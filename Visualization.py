# Imports
import matplotlib.pyplot as plt
import csv
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
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from itertools import combinations
import HigherOrderProducer 
import LogisticSubsetSelection
from scipy import stats

data = pd.read_csv('./data/TrainData_Labeled.csv')
# print("Length of data: ", len(data))
# data = data.drop_duplicates(keep='first')
# print("Length of UNIQUE data: ", len(data))


# Baseline performance for random state 27 before cleaning data:
# Random Forest: 64/38/64
# XGB: 62/36/62

# Dropping Class 7 two outliers:
# RF: 66/40/65 
# XGB: 64/44/63

# Dropping Class 7 two outliers and Class 6 outlier:
# RF: 69/33/67
# XGB: 66/33/65

# Remove outliers - this DOES improve performance :)
# print(data.iloc[603])
# print(data.iloc[329])
# data = data.drop(329) # Class 7 high sulfate
# data = data.drop(603) # Class 7 high sulfate
# data = data.drop(399) # Class 6 high free sulfur dioxide

# points = data[data['free sulfur dioxide'] > 60] # Filter out on criteria
# print(points)

# plt.scatter(data['total sulfur dioxide'], data['pH'])
# plt.show()
# plt.figure()

test_data = pd.read_csv('./data/TestData.csv')

# Performs classification with given model and data.
# @param model: the SKLearn model to use for predictions
# @param data: The data to use for training and testing
# @param print_csv: whether or not the output should be printed to a CSV file
def classify(model, data, print_csv=False):
    # Convert data from Pandas DataFrame to np array for X and y
    X = data[data.columns[:11]]
    y = data[data.columns[11]]

    # Partition data into train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=27)
    
    # Combine the train data with its labels so we can split up by class and remove outliers
    train_full = X_train
    train_full['label'] = y_train
    print(train_full)
    filtered_data = None
    # Separate data into classes
    for i in range(3, 9):
        classi = X_train[train_full['label'] == i]
        z = np.abs(stats.zscore(classi))
        classi_o = classi[(z < 3).all(axis=1)]
        print(f'Outliers in class {i}: {len(classi_o)}')

        # f = classi['alcohol']
        # classi = classi[f.between(f.quantile(.15), f.quantile(.85))]
        # print("Outliers removed: ", len(classi))
        # print("All: ", len(f))
        # if filtered_data is None:
        #     filtered_data = classi
        # else:
        #     filtered_data = filtered_data.append(classi)
        # l = classi[(np.abs(stats.zscore(classi)) < 3).all(axis=1)]
        # print(f"Length of class {i}: {len(l)}")
    print(filtered_data)

    X_train = filtered_data[filtered_data.columns[:11]]
    y_train = filtered_data[filtered_data.columns[11]]
    

    # Classify with given model and print report
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    # Rounds predictions to nearest value for cases like XGBoost
    pred = [round(value) for value in pred]

    # print(classification_report(y_test, pred))
    scores = cross_val_score(model, X_test, y_test, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))

    print(classification_report(y_test, pred))
    scores = cross_val_score(model, X_test, y_test, cv=5)
    # This particular metric is probably not as useful as the whole report, printed below
    #print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
    # mae = mean_absolute_error(pred, y_test)
    # print("Mean absolute error: %f" % (mae))
    # acc_score = accuracy_score(y_test, pred)
    # print("Accuracy score: %2f\n" %(acc_score))
    testdata = np.asarray(data)[:, :11]
    pred = model.predict(testdata)
    pred = [round(value) for value in pred]

    # Do it on the REAL test data
    realtest = np.asarray(test_data)[:, :11]
    realpred = model.predict(realtest)
    realpred = [round(value) for value in realpred]
    # Print to CSV
    if print_csv:
        with open('result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in realpred:
                writer.writerow([row])

def performPlots(data):
    dataToPlot = []
    for i in range(3, 9):
        is_class = data['label'] == i
        dataToPlot.append(data[is_class])
    #data to plot: each index is a diff class, has 12 cols
    print(len(dataToPlot))
    choices = list(range(11))
    subsets = list(combinations(choices, 2))
    strToSave = ""
    for sub in subsets:
        print(sub)
        plt.figure()
        for c in range(0, len(dataToPlot)):
            is_x = dataToPlot[c].iloc[:,sub[0]]
            is_y = dataToPlot[c].iloc[:,sub[1]]
            is_class = ['label'] == i
            color = colorFromIndex(c+3)
            plt.plot(is_x, is_y, 'x', label=c+3, color=color)
        plt.legend()
        strToSave = str(sub[0]) + "_vs_" + str(sub[1])
        plt.savefig(strToSave)

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
    


# ylabels = np.asarray(data['label'])
# ylabels = [round(value) for value in ylabels]
# plt.hist(ylabels, bins=6)
# plt.ylabel('Count')
# plt.xlabel('Label')
# plt.show()

# Classify with Stochastic Gradient Descent & print report
sgd = SGDClassifier(penalty=None)
print("Stochastic Gradient Descent classifier results:")
#classify(sgd, data)

# Classify with K Nearest Neighbors & print report
knn = KNeighborsClassifier(n_neighbors = 5)
print("K Nearest Neighbors classifier results:")
#classify(knn, data)

# Classify with Random Forest Classifier & print report
rfc = RandomForestClassifier(n_estimators=500, random_state=27)
print("Random Forest classifier results:")
classify(rfc, data)

#performPlots(data)
# Classify with EXTREME GRADIENT BOOOOSTING & print report
xgb = XGBRegressor(n_estimators=750, n_jobs=4, objective='reg:squarederror', tree_method='hist')
print("EXTREME Gradient Boosting results:")
#classify(xgb, data, print_csv=False)
