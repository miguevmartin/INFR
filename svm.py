# -*- coding: utf-8 -*-
"""
Modified Sept 2019

@author: mvm for INFR3700 and MITS6800 (based on Geron)
"""

import pandas as pd
sal_data = pd.read_csv('attributes_vs_salary.dat')

# Create training/testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(sal_data, test_size=0.2)#, random_state=42)
train_labels = train['Income ($K/year)'] >=100 #labels are Boolean
train_data = train.iloc[:,1:3] #take only YofEd and Age columns

# Graph <100 and >=100 with different shapes
import matplotlib.pyplot as plt
over100_yoe = train.loc[train['Income ($K/year)'] >= 100].iloc[:,1]
under100_yoe = train.loc[train['Income ($K/year)'] < 100].iloc[:,1]
over100_age = train.loc[train['Income ($K/year)'] >= 100].iloc[:,2]
under100_age = train.loc[train['Income ($K/year)'] < 100].iloc[:,2]
plt.scatter(over100_yoe, over100_age, color='r', label='over $100K', marker='^', s = 70)
plt.scatter(under100_yoe, under100_age, color='b', label='under $100K', marker='v', s = 70)
plt.legend()
plt.xlabel("years of education")
plt.ylabel("age")

# Fit an SVM
from sklearn.svm import SVC
svm_clf = SVC(kernel='poly', degree=4)#defaults: kernel="rbf", C=1.0)
svm_clf.fit(train_data, train_labels)
pred = svm_clf.predict(train_data)
print("Number of mislabeled points out of a total %d points: %d. Accuracy: %f" 
     % (len(train_data),(train_labels != pred).sum(), 1-(train_labels != pred).sum()/len(train_data)))
from sklearn.metrics import confusion_matrix
print('Confusion matrix (TN,FP/FN,TP):\n', confusion_matrix(train_labels, pred))
from sklearn.metrics import precision_score, recall_score
print('Precision:', precision_score(train_labels, pred))
print('Recall:', recall_score(train_labels, pred))
from sklearn.metrics import f1_score
print('F1 Score:', f1_score(train_labels, pred))

# Graph contour of our classification model 
import numpy as np
x0s = np.linspace(-1, 31, 100)
x1s = np.linspace(0, 60, 100)
x0, x1 = np.meshgrid(x0s, x1s)
X = np.c_[x0.ravel(), x1.ravel()]
y_pred = svm_clf.predict(X).reshape(x0.shape)
plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
plt.show()