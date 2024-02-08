import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

path = "C:/Users/liork/Downloads/machine.data"

Names = ['Vendor', 'Model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP', 'Target']
data = pd.read_csv(path, names = Names)
data.head()

X = data.drop(columns=['Vendor','Model','ERP', 'Target'])
Y = data['Target']
print(X)
print(Y)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X_scal = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scal, Y, test_size = 0.30)

from sklearn.neighbors import KernelDensity

for i in range(1,11):
    kde = KernelDensity(bandwidth=i, kernel='cosine')
    kde.fit(X_train)
    print(kde.score(X_test))