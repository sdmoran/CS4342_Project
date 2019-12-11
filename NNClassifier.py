from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier

data = pd.read_csv('./data/TrainData_Labeled.csv')

# Convert data from Pandas DataFrame to np array for X and y
X = np.asarray(data)[:, :11]
y = np.asarray(data)[:, 11]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

# Build Keras model
n_cols = X.shape[1]

def make_model():
    model = Sequential()
    # Input layer
    model.add(Dense(11, activation='exponential', input_shape=(n_cols,)))
    # Hidden layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=make_model, epochs=20, batch_size=5, verbose=0)
kfold = KFold(n_splits=4, shuffle=True)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))