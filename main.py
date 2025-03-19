# import numpy as np
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

sns.countplot(x='type', data=data)

sns.barplot(x='type', y='amount', data=data)

print("data['isFraud'].value_counts()")
print(data['isFraud'].value_counts())
print(data['isFraud'].value_counts().keys())
print(data['isFraud'].value_counts().items())
print()

exit(0)
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
exit(0)

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
for i in range(len(models)):
    models[i].fit(X_train, y_train)
    print(f'{models[i]} : ')

    train_predictions = models[i].predict_proba(X_train)[:, 1]
    temp_train  =  ras(y_train, train_predictions)
    print('Training Accuracy : ', temp_train)

    y_predictions = models[i].predict_proba(X_test)[:, 1]
    temp_val = ras(y_test, y_predictions)
    print('Validation Accuracy : ', temp_val)
    print("="*30)
    if temp_val > validation_accuracy and temp_train > training_accuracy:
        validation_accuracy = temp_val
        training_accuracy = temp_train
        model_num = i


print()
print(f'Winner: {models[model_num]}')
print('Training Accuracy : ', training_accuracy)
print('Validation Accuracy : ', validation_accuracy)
cm = ConfusionMatrixDisplay.from_estimator(models[model_num], X_test, y_test)

cm.plot(cmap='Blues')

plt.show()

# save
with open('model.pkl','wb') as f:
    pickle.dump(models[model_num],f)


# # load
# with open('model.pkl', 'rb') as f:
#     clf2 = pickle.load(f)