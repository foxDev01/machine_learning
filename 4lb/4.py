import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
# from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd
#логистич регресия
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
X = dataset.iloc[:, :-1] #выбор вс
Y = dataset.iloc[:,7]
#print(dataset.shape)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.40,  random_state = 1)

scaler  = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for i in range(1,11):
    print(i)
    model = LogisticRegression(C=i,random_state=800, max_iter=3000).fit(X_train,Y_train)
    model.predict(X_test)
    #print(model.predict_proba(X_test))
    print(model.score(X_test, Y_test))

Y_test=Y_test.to_numpy()
fig = plt.figure(figsize=(10,10), facecolor='#87CEFA')
for j in range(0,6):
    d = fig.add_subplot(3,2,j+1)#сетка 
    logistreg = LogisticRegression(C=1, solver='liblinear', random_state=800)
    logistreg = logistreg.fit(X_train[:,[j+1,0]], Y_train)
    plot_decision_regions(X_test[:,[j+1,0]], Y_test, clf=logistreg, legend=2)
    
    plt.xlabel(dataset.columns[j+1],labelpad=10)
    plt.ylabel(dataset.columns[0])
    plt.tight_layout(h_pad=3)   

plt.show()