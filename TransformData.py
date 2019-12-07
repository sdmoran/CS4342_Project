# This file is for transforming the data from its original form into one more usable by Pandas, Seaborn, etc.
# This is mostly utility stuff.

import csv
        
# Loads a CSV file from the given path
# @param path: the path of the file to open
# @param Labels: A boolean indicating whether or not the first row gives the column labels.
def load_from_csv(path, labels=False):
    contents = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        if labels:
            labels = next(csv_reader)
        for row in csv_reader:
            contents.append([float(i) for i in row])
        if labels:
            return (contents, labels)
        return contents

# Load train data and names of features from CSV file
data, feature_names = load_from_csv('./data/winequality-redTrainData.csv', labels=True)

# Load labels for datapoints
labels = load_from_csv('./data/winequality-redTrainLabel.csv')

# Load test data. Like train data, also has labels.
test_data, test_features = load_from_csv('./data/winequality-redTestData.csv', labels=True)


# Combine data with labels
labeled_data = [data[i] + labels[i] for i in range(len(data))]

with open('TrainData_Labeled.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(feature_names)
    for row in labeled_data:
        writer.writerow(row)

# Write test data to more readable format
with open('TestData.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(feature_names)
    for row in test_data:
        writer.writerow(row)