import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score
from sklearn.svm import SVC
import pandas as pd
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 
#опорных векторов 
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
X = dataset.iloc[:, :-1] #выбор 
Y = dataset.iloc[:,7]
print(dataset.shape)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.40,  random_state = 1)
print(X)
scaler  = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Y_test=Y_test.to_numpy()

fig = plt.figure(figsize=(10,10), facecolor='#87CEFA')

for i in range(0,6):
    d = fig.add_subplot(3,2,i+1)#сетка 
    classifier = SVC(C=1.0, kernel = 'sigmoid', random_state = 80, gamma=1,  ) #'linear', 'poly', 'rbf', 'sigmoid'
    classifier.fit(X_train[:,[i+1,0]], Y_train)
    
    plot_decision_regions(X_test[:,[i+1,0]], Y_test, clf=classifier, legend=2)
    plt.xlabel(dataset.columns[i+1],labelpad=10)            #название оси
    plt.ylabel(dataset.columns[0])                          #название оси 
    plt.tight_layout(h_pad=3)                               #регулировка растояние между графиками
y_predict = classifier.predict(X_test[:,[i+1,0]])
print(recall_score(Y_test, y_predict, average=None))
print('число ошибочных персептронов: %d' % (Y_test != y_predict).sum())
pred_svc2 = classifier.predict(X_test[:,[i+1,0]])
print(classification_report(Y_test, pred_svc2))

plt.show()