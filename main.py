# import numpy as np
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay

import pickle

show_charts: bool = False

data = pd.read_csv('new_file.csv')
print("data.head()")
print(data.head())
print()
print("data.info()")
print(data.info())
print()
print("data.describe()")
print(data.describe())
print()
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (data.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (data.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

if show_charts:
    sns.countplot(x='type', data=data)

    sns.barplot(x='type', y='amount', data=data)

print()
print("Fraud value_counts")
print("=" * 30)
is_fraud = json.loads(data['isFraud'].value_counts().to_json())

total_count = is_fraud["0"] + is_fraud["1"]
false_percent = is_fraud["0"] / total_count * 100
true_percent = is_fraud["1"] / total_count * 100

print(f"total_count: {total_count}")
print()
print(f"false count: {is_fraud['0']}, {false_percent:.2f}%")
print(f"true count: {is_fraud['1']}, {true_percent:.2f}%")

print("=" * 30)

if show_charts:
    plt.figure(figsize=(15, 6))
    sns.histplot(data['step'], bins=50, kde=True)

    plt.figure(figsize=(12, 6))
    sns.heatmap(data.apply(lambda x: pd.factorize(x)[0]).corr(),
                cmap='BrBG',
                fmt='.2f',
                linewidths=2,
                annot=True)

type_new = pd.get_dummies(data['type'], drop_first=True)
data_new = pd.concat([data, type_new], axis=1)
print("data_new.head()")
print(data_new.head())
print()
X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']

print("X.shape, y.shape")
print(X.shape, y.shape)
print()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.31, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.999, random_state=24)

print("X_train, X_test, X_validation, y_train, y_test,  y_validation")
print(X_train.shape, X_test.shape, X_validation.shape, y_train.shape, y_test.shape,  y_validation.shape)
print("*" * 30)
print()
X_validation = pd.concat([X_validation, y_validation], axis=1)
X_validation.to_csv("sample.csv")

# LogisticRegression(max_iter=200) max_iter defaults to 100
models = [LogisticRegression(max_iter=200), XGBClassifier(),
		RandomForestClassifier(n_estimators=7,
								criterion='entropy',
								random_state=7)]

training_accuracy = 0
validation_accuracy = 0
model_num = -1
temp_train = 0
temp_val = 0

print()
print("training models, please wait")
print()
for i in range(len(models)):
    models[i].fit(X_train, y_train)
    print(f'{models[i]} : ')

    train_predictions = models[i].predict_proba(X_train)[:, 1]
    temp_train  =  ras(y_train, train_predictions)
    print(f'Training Accuracy: {temp_train*100:.2f}%')

    y_predictions = models[i].predict_proba(X_test)[:, 1]
    temp_val = ras(y_test, y_predictions)
    print(f'Validation Accuracy: {temp_val*100:.2f}%')
    print("-"*30)
    print()
    if temp_val > validation_accuracy and temp_train > training_accuracy:
        validation_accuracy = temp_val
        training_accuracy = temp_train
        model_num = i
print("training done")

print()
print("-="*50)
print(f'Winner: {models[model_num]}')
print(f'Training Accuracy: {training_accuracy*100:.2f}%')
print(f'Validation Accuracy: {validation_accuracy*100:.2f}%')
print("-="*50)
print()
if show_charts:
    cm = ConfusionMatrixDisplay.from_estimator(models[model_num], X_test, y_test)

    cm.plot(cmap='Blues')

    plt.show()

print("sample data found in sample.csv")
print()

# save
with open('model.pkl','wb') as f:
    pickle.dump(models[model_num],f)

print("model saved to model.pkl")
print()

# # load
# with open('model.pkl', 'rb') as f:
#     clf2 = pickle.load(f)