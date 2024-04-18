# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN

warnings.filterwarnings(action='ignore')

"""Importing the dataset"""

from google.colab import drive
drive.mount('/content/drive')

"""Reading the dataset"""

df = pd.read_csv('/content/drive/MyDrive/dataset/cicddos2019_dataset.csv')

df.head(15)

"""Exploring the dataset"""

df.shape

df.info()

df.describe()

df.nunique()

"""Assigning Binary Values"""

df_binary = df.Class.map(lambda a: 0 if a == 'Benign' else 1)
df['Binary_state'] = df_binary

df.head(15)

"""Checking for Missing values"""

df.isnull().sum()

"""Checking for duplicate values"""

df.duplicated().sum()

df = df.drop(['Label','Class'], axis = 1 )

print(df.head(15))

"""Splitting the dataset into training and testing"""

X = df.drop('Binary_state', axis=1)
y = df['Binary_state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

"""Scaling the dataset"""

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

"""Defining the Evaluate function"""

def Evaluate(Model_Name, Model_Abb, X_test, y_test):
    pred_value = Model_Abb.predict(X_test)
    Accuracy = metrics.accuracy_score(y_test, pred_value)
    Precision = metrics.precision_score(y_test, pred_value)
    F1_score = metrics.f1_score(y_test, pred_value)
    Recall = metrics.recall_score(y_test, pred_value)
    Confusion_mat = metrics.confusion_matrix(y_test, pred_value)

    print('--------------------------------------------------\n')
    print('The {} Model Accuracy   = {}\n'.format(Model_Name, np.round(Accuracy, 3)))
    print('The {} Model Precision  = {}\n'.format(Model_Name, np.round(Precision, 3)))
    print('The {} Model F1 Score   = {}\n'.format(Model_Name, np.round(F1_score, 3)))
    print('The {} Model Recall     = {}\n'.format(Model_Name, np.round(Recall, 3)))
    print('The {} Model Confusion Matrix     = {}\n'.format(Model_Name, np.round(Confusion_mat, 3)))
    print('--------------------------------------------------\n')

"""Random Forest"""

RF= RandomForestClassifier()
RF.fit(X_train_scaled, y_train)
Evaluate('Random Forest Classifier',RF, X_test_scaled, y_test)

"""K-Nearest Neighbor"""

KNN = KNeighborsClassifier()
KNN.fit(X_train_scaled, y_train)
Evaluate('KNN', KNN, X_test_scaled, y_test)

"""Logistic Regression"""

LR = LogisticRegression(max_iter=1000)
LR.fit(X_train_scaled, y_train)
Evaluate('LR', LR, X_test_scaled, y_test)

"""Undersampling the dataset"""

y.value_counts()

y.value_counts().plot.pie(autopct='%.2f')

#Random Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1)
X_res_u, y_res_u = rus.fit_resample(X,y)

ax = y_res_u.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Under-sampling")

#Class distribution
y_res_u.value_counts()

"""Splitting the new dataset into training and testing (Undesampled)"""

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_res_u, y_res_u, test_size=0.3, random_state=42)

print("X_train_u shape:", X_train_u.shape)
print("X_test_u shape:", X_test_u.shape)
print("y_train_u shape:", y_train_u.shape)
print("y_test_u shape:", y_test_u.shape)

X_train_u_scaled = scaler.fit_transform(X_train_u)
X_test_u_scaled = scaler.fit_transform(X_test_u)

print("X_train_u_scaled shape:", X_train_u_scaled.shape)
print("X_test_u_scaled shape:", X_test_u_scaled.shape)

"""Random Forest"""

RF.fit(X_train_u_scaled, y_train_u)
Evaluate('Random Forest Classifier',RF, X_test_u_scaled, y_test_u)

"""K-Nearest Neighbour"""

KNN.fit(X_train_u_scaled, y_train_u)
Evaluate('KNN', KNN, X_test_u_scaled, y_test_u)

"""Logistic Regression"""

LR.fit(X_train_u_scaled, y_train_u)
Evaluate('LR', LR,X_test_u_scaled, y_test_u)

"""Oversampling the dataset"""

from imblearn.over_sampling import RandomOverSampler

#ros = RandomOverSampler(sampling_strategy=1) # Float
ros = RandomOverSampler(sampling_strategy="not majority") # String
X_res_o, y_res_o = ros.fit_resample(X, y)

ax = y_res_o.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling")

y_res_o.value_counts()

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_res_o, y_res_o, test_size=0.3, random_state=42)

print("X_train_o shape:", X_train_o.shape)
print("X_test_o shape:", X_test_o.shape)
print("y_train_o shape:", y_train_o.shape)
print("y_test_o shape:", y_test_o.shape)

X_train_o_scaled = scaler.fit_transform(X_train_o)
X_test_o_scaled = scaler.fit_transform(X_test_o)

print("X_train_o_scaled shape:", X_train_o_scaled.shape)
print("X_test_o_scaled shape:", X_test_o_scaled.shape)

RF.fit(X_train_o_scaled, y_train_o)
Evaluate('Random Forest Classifier',RF, X_test_o_scaled, y_test_o)

KNN.fit(X_train_o_scaled, y_train_o)
Evaluate('KNN', KNN, X_test_o_scaled, y_test_o)

LR.fit(X_train_o_scaled, y_train_o)
Evaluate('LR', LR,X_test_o_scaled, y_test_o)

"""HyperParameter Tuning using HalvingRandomSearchCV (Original Dataset)


"""

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
#HalvingRandomSearchCV on Random Forest
paramGrid_rf = {
    'n_estimators':[100,200,300],
    'max_depth': [None,5,10],
    'min_samples_split': [2,5,10],
    'min_samples_leaf' : [1,2,4]
}

halving_search_rf = HalvingRandomSearchCV(estimator=RF, param_distributions = paramGrid_rf,factor = 5, cv=3)
halving_search_rf.fit(X_train_scaled, y_train)

best_rf = halving_search_rf.best_estimator_
best_params_rf = halving_search_rf.best_params_

print("Best Parameters for Random Forest: ", best_params_rf)
Evaluate('Random Forest Classifier', best_rf, X_test_scaled, y_test)

#HalvingRandomSearchCV on K-Nearest Neighbor
paramGrid_knn = {
    'n_neighbors':[3,5,7],
    'weights':['uniform','distance'],
    'metric' :['euclidiean','manhattan']
}

halving_search_knn = HalvingRandomSearchCV(estimator=KNN, param_distributions = paramGrid_knn, factor = 2, cv=3)
halving_search_knn.fit(X_train_scaled, y_train)

best_knn = halving_search_knn.best_estimator_
best_params_knn = halving_search_knn.best_params_

print("Best Parameters for K-Nearest Neighbor: ", best_params_knn)
Evaluate('KNN', best_knn, X_test_scaled, y_test)

#HalvingRandomSearchCV on Logistic Regression
paramGrid_lr = {
    'C':[0.01,0.1,1],
    'penalty': ['l1','l2'],
    'solver':['liblinear','saga']
}

halving_search_lr = HalvingRandomSearchCV(estimator=LR, param_distributions = paramGrid_lr,factor=2, cv=3)
halving_search_lr.fit(X_train_scaled, y_train)

best_lr = halving_search_lr.best_estimator_
best_params_lr = halving_search_lr.best_params_

print("Best Parameters for Logistic Regression: ", best_params_lr)
Evaluate('LR', best_lr, X_test_scaled, y_test)

"""HyperParameter Tuning using HalvingRandomSearchCV (Undersampled Dataset)

---


"""

random_search_rf_u = HalvingRandomSearchCV(estimator=RF, param_distributions = paramGrid_rf,factor=2, cv=3)
random_search_rf_u.fit(X_train_u_scaled, y_train_u)

best_rf = random_search_rf_u.best_estimator_
best_params_rf = random_search_rf_u.best_params_

print("Best Parameters for Random Forest: ", best_params_rf)
Evaluate('Random Forest Classifier', best_rf, X_test_scaled, y_test)

random_search_knn_u = HalvingRandomSearchCV(estimator=KNN, param_distributions = paramGrid_knn,factor=2, cv=3)
random_search_knn_u.fit(X_train_u_scaled, y_train_u)

best_knn_u = random_search_knn_u.best_estimator_
best_params_knn_u = random_search_knn_u.best_params_

print("Best Parameters for K-Nearest Neighbors (Undersampled): ", best_params_knn_u)
Evaluate('KNN', best_knn_u, X_test_u_scaled, y_test_u)

random_search_lr_u = HalvingRandomSearchCV(estimator=LR, param_distributions = paramGrid_lr,factor=2, cv=3)
random_search_lr_u.fit(X_train_u_scaled, y_train_u)

best_lr_u = random_search_lr_u.best_estimator_
best_params_lr_u = random_search_lr_u.best_params_

print("Best Parameters for Logistic Regression: ", best_params_lr_u)
Evaluate('LR', best_lr_u, X_test_u_scaled, y_test_u)

"""HyperParameter Tuning using HalvingRandomSearchCV (Oversampled Dataset)"""

random_search_rf_o = HalvingRandomSearchCV(estimator=RF, param_distributions = paramGrid_rf, factor=2, cv=3)
random_search_rf_o.fit(X_train_o_scaled, y_train_o)

best_rf_o = random_search_rf_o.best_estimator_
best_params_rf_o = random_search_rf_o.best_params_

print("Best Parameters for Random Forest (Oversampled): ", best_params_rf_o)
Evaluate('Random Forest Classifier', best_rf_o, X_test_o_scaled, y_test_o)

random_search_knn_o = HalvingRandomSearchCV(estimator=KNN, param_distributions = paramGrid_knn, factor=2, cv=3)
random_search_knn_o.fit(X_train_o_scaled, y_train_o)

best_knn_o = random_search_knn_o.best_estimator_
best_params_knn_o = random_search_knn_o.best_params_

print("Best Parameters for K-Nearest Neighbors (Oversampled): ", best_params_knn_o)
Evaluate('KNN', best_knn_o, X_test_o_scaled, y_test_o)

random_search_lr_o = HalvingRandomSearchCV(estimator=LR, param_distributions = paramGrid_lr, factor=2, cv=3)
random_search_lr_o.fit(X_train_o_scaled, y_train_o)

best_lr_o = random_search_lr_o.best_estimator_
best_params_lr_o = random_search_lr_o.best_params_

print("Best Parameters for Logistic Regression (Oversampled): ", best_params_lr_o)
Evaluate('LR', best_lr_o, X_test_o_scaled, y_test_o)
