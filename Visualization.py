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
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

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
    # Rounds predictions to nearest value for cases like XGBoost
    pred = [round(value) for value in pred]
    #print(classification_report(y_test, pred))
    #scores = cross_val_score(model, X_test, y_test, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
    mae = mean_absolute_error(pred, y_test)
    print("Mean absolute error: %f" % (mae))
    acc_score = accuracy_score(y_test, pred)
    print("Accuracy score: %2f\n" %(acc_score))
    
# Heatmap correlations between different variables.
plt.figure(figsize=(12, 12))
sns.heatmap(data=data.corr(), annot=True)
#plt.show()

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

# Classify with EXTREME GRADIENT BOOOOSTING & print report
xgb = XGBRegressor(n_estimators=750, learning_rate=0.05, n_jobs=4, objective='reg:squarederror', tree_method='hist')
print("EXTREME Gradient Boosting results:")
classify(xgb, data)
