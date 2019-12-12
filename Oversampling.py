from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBRegressor

test_data = pd.read_csv('./data/TestData.csv')

def classify(model, X_train, y_train, X_test, y_test, print_csv=True):
   
    # Classify with given model and print report
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # Rounds predictions to nearest value for cases like XGBoost
    pred = [round(value) for value in pred]

    print(classification_report(y_test, pred))

    # Do it on the REAL test data
    realtest = test_data[test_data.columns[:11]]
    realpred = model.predict(realtest)
    realpred = [round(value) for value in realpred]
    # Print to CSV
    if print_csv:
        with open('result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in realpred:
                writer.writerow([row])

    return pred

data = pd.read_csv('./data/TrainData_Labeled.csv')


#X_resampled['label'] = y_resampled

#print(X_resampled)



# Classify with Random Forest Classifier on original data & print report
rfc = RandomForestClassifier(n_estimators=500)
print("Random Forest classifier results:")

# Partition data into train and test samples
X = data[data.columns[:11]]
y = data[data.columns[11]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, )

# Resample from training data, oversampling underrepresented classes
X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X_train, y_train)
X_resampled.append(X_train)
y_resampled.append(y_train)

classes = {'3.0':0, '4.0':0, '5.0':0, '6.0': 0, '7.0': 0, '8.0': 0}
for pt in y_resampled:
    classes[str(pt)] += 1

# Print classification for regular data
classify(rfc, X_train, y_train, X_test, y_test)

print("Resampling classifier results:")
rfc2 = RandomForestClassifier(n_estimators=500)

classify(rfc2, X_resampled, y_resampled, X_test, y_test)

# Try it with XGB
print("XGB classifier results:")  
xgb = XGBRegressor(n_estimators=750, n_jobs=4, objective='reg:squarederror', tree_method='hist')
classify(xgb, X_train, y_train, X_test, y_test)


# OK it seems like this one might be actually good actually for real for real though?
print("XGB Resampling classifier results:")
xgb2 = XGBRegressor(n_estimators=750, n_jobs=4, objective='reg:squarederror', tree_method='hist')
#xgb_resample_results = classify(xgb2, X_resampled, y_resampled, X_test, y_test)
xgb_resample_results = classify(xgb2, X_resampled, y_resampled, X_test, y_test, print_csv=True)   

# print(len(xgb_resample_results), len(y_test))

# correct = 0
# for i in range(len(y_test)):
#     if xgb_resample_results[i] == y_test.iloc[i]:
#         correct += 1

# print(f"Correct: {correct / len(y_test)}")

# What if we trained a bunch of binary classification regressors? Split them up into 3, 4, or 5 vs 6, 7, 8
# Then do 6 or 7 vs 8 and 4 or 5 vs 3
# Then do 6 vs 7 and 4 vs 5
# tadahhh?