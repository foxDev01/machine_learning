import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
# from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd
#деревья
url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 
names = [
    "площадь",                 
    "периметр",             
    "компактность", 
    "длина",         
    "ширина",          
    "асимметрия",
    "Длина канавки ядра",        
    "Сорт"            
] #название атрибутов 

dataset = pd.read_csv(url, names=names) #чтение файла с данными 
dataset.head()
X = dataset[["площадь","компактность"]]
X = X.values               
Y = dataset[["Сорт"]]
Y = Y.values
#только сорт
#print(dataset.shape)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=1
)
X_combined=np.vstack((X_train,X_test))
Y_combined=np.vstack((Y_train,Y_test))
#создание
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion="gini", 
                              max_depth=7, 
                              random_state=0) #criterion="entropy" джинни 
tree.fit(X_train, Y_train)
scores = cross_val_score(tree, X, Y, cv=5)
print('DecisionTreeClassifier:')

print('Score: ',scores.mean()) 
#отрисовка 
from sklearn.tree import export_graphviz

export_graphviz(
    tree, out_file="treeDecision.dot", feature_names=["площадь", "компактность"]
)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( criterion="gini",
                                max_depth=7, 
                                random_state=0)
forest.fit(X_train, Y_train) 
estimator = forest.estimators_[5]

scores = cross_val_score(forest, X, Y, cv=5)
print("RandomForestClassifier:")
print('Score: ',scores.mean()) 

dot_data = export_graphviz(estimator, out_file="treeRandom.dot", max_depth=2,feature_names=["площадь", "компактность"], filled=True, rounded=True,)



# случайный лес меняется класс который работает с ним ctrl+shift+v