import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

if lgb.compat.MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
else:
    raise ImportError('You need to install matplotlib for plot_example.py.')

print('Loading data...')
# load or create your dataset
data = pd.read_csv('./data/TrainData_Labeled.csv')

# Convert label to (0, 6) scale so lightgbm knows what to do with it
data['label'] -= 3
print(data)

X = np.asarray(data)[:, :11]
y = np.asarray(data)[:, 11]

# Partition data into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, )

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'objective': 'multiclass',
    'num_class': 6,
    'num_leaves': 20, # 20 best so far
    'learning_rate': 0.005,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.9,
    'metric': 'multiclassova',
    'verbose': 0
}

evals_result = {}  # to record eval results for plotting

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                early_stopping_rounds=300,
                valid_sets=[lgb_train, lgb_test],
                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],
                categorical_feature=[21],
                evals_result=evals_result,
                verbose_eval=10)

pred = gbm.predict(X_test)
converted_preds = []
for x in pred:
    converted_preds.append(np.argmax(x))
print(converted_preds)

print("Classification report: ", classification_report(y_test, converted_preds))

print('Plotting metrics recorded during training...')
ax = lgb.plot_metric(evals_result)
plt.show()

print('Plotting feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()

# print('Plotting split value histogram...')
# ax = lgb.plot_split_value_histogram(gbm, feature='f26', bins='auto')
# plt.show()

print('Plotting 54th tree...')  # one tree use categorical feature to split
ax = lgb.plot_tree(gbm, tree_index=53, figsize=(15, 15), show_info=['split_gain'])
plt.show()

print('Plotting 54th tree with graphviz...')
graph = lgb.create_tree_digraph(gbm, tree_index=53, name='Tree54')
graph.render(view=True)