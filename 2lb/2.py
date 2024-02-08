import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

# from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd

url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 

names = [
    "площадь A",                 
    "периметр P",             
    "компактность C ", 
    "длина",         
    "ширина",          
    "асимметрия",
    "Длина канавки ядра",        
    "Сорт"            
] #название атрибутов 

dataset = pd.read_csv(url, names=names) #чтение файла с данными 
dataset.head()
X = dataset.iloc[:, :-1] #выбор вс
Y = dataset.iloc[:,7]
#print(dataset.shape)
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.40,  random_state = 1)
scaler  = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#выбор метрик 
for j in range(4):
    print( '\n' +(metrics[j-1]))
    metricIndex = j-1
    #resault.append(metrics[j-1])
#выыод расчета с соответствующими  метриками 
    for i in range(10):
        classifier = KNeighborsClassifier(n_neighbors = i+1, metric = metrics[metricIndex])
        classifier.fit(X_train,Y_train)
        Y_pred = classifier.predict(X_test)
        score = cross_val_score(classifier, X_test, Y_pred)
        print('коэффициент - {}, '.format(i+1), end = ' ')
        print('точность {}%'.format(int(round(np.mean(score), 2) * 100)))
        
