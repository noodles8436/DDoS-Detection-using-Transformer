from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import random

random.seed(42)

DATASET_ROOT = "./dataset-preprocessed"
dataList = os.listdir("./dataset-preprocessed")
random.seed(42)
random.shuffle(dataList)

trainList = dataList[:int(len(dataList) * 0.8)]
valList = dataList[int(len(dataList) * 0.8):int(len(dataList) * 0.9)]
testList = dataList[int(len(dataList) * 0.9):]


def getDataset(_datalist):
    result_x = []
    result_y = []
    for filename in _datalist:
        sample = np.load(os.path.join("./dataset_preprocessed", filename))
        data_x: np.ndarray = sample['x']
        data_y: np.ndarray = sample['y']

        data_x = data_x.reshape((-1,))
        data_y = data_y.reshape((-1,))

        result_x.append(data_x)
        result_y.append(data_y)

    result_y = np.array(result_y).reshape(-1, 1)

    return result_x, result_y

x_train, y_train = getDataset(trainList)
x_val, y_val = getDataset(valList)
x_test, y_test = getDataset(testList)

model_name = 'Logistic'
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(model_name, 'precision', precision)
print(model_name, 'recall', recall)
print(model_name, 'auc', auc)

model_name = 'DecisionTree'
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(model_name, 'precision', precision)
print(model_name, 'recall', recall)
print(model_name, 'auc', auc)

model_name = 'SVM'
model = SVC(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(model_name, 'precision', precision)
print(model_name, 'recall', recall)
print(model_name, 'auc', auc)

model_name = 'RandomForest'
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(model_name, 'precision', precision)
print(model_name, 'recall', recall)
print(model_name, 'auc', auc)
