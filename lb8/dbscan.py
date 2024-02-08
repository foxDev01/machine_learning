
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
import seaborn as sns

url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 

names = ["площадь","периметр","компактность", "длина","ширина","асимметрия","длина канавки ядра","сорт"] #название атрибутов 
target_names = { 1:"Кама" , 2:"Роза" , 2:"Канадка" }
dataset = pd.read_csv(url, names=names)
dataset.head()
X = dataset.iloc[:,0:7] 
y = dataset['сорт'].values
print(X)

from sklearn.cluster import DBSCAN
db = DBSCAN()
Y_preds = db.fit_predict(X)
print(Y_preds)
plt.scatter(X[Y_preds==0,0],X[Y_preds==0,-1], c = 'red', marker="^", s=50)
plt.scatter(X[Y_preds==-1,0],X[Y_preds==-1,-1], c = 'green', marker="^", s=50)
plt.scatter(X[Y_preds==2,0],X[Y_preds==2,-1], c = 'blue', marker="^", s=50)
plt.legend()
plt.show()