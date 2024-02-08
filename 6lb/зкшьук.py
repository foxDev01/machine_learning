import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

clfTre = tree.DecisionTreeClassifier(max_depth=2)
clfTre.fit(X_train, Y_train)

dot_data = tree.export_graphviz(clfTre, out_file="resume.dot", max_depth=2, feature_names=list(X.columns), filled=True, rounded=True,)
valgTre = graphviz.Source(dot_data) 

import pydotplus
from IPython.display import Image

graph = pydotplus.graphviz.graph_from_dot_file("resume.dot")
graph.write_png("dshjsf.png")


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, Ytrain)

estimator = clf.estimators[5]

from sklearn.tree import export_graphviz

dot_data = export_graphviz(estimator, out_file="resume2.dot", max_depth=2, feature_names=list(X.columns), filled=True, rounded=True,)
valgTre = graphviz.Source(dot_data) 

graph = pydotplus.graphviz.graph_from_dot_file("resume2.dot")
graph.write_png("dshbvgjsf.png")