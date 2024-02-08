# Создание основного объекта персептрона.
class Perceptron(object):
    #Начальная скорость обучения и количество итераций.
    def __init__(self, Learn_Rate=0.5, Iterations=10):
        self.learn_rate = Learn_Rate
        self.Iterations = Iterations
        self.errors = []
        self.weights = np.zeros(1 + X.shape[1])
    
    # Определение метода подгонки для обучения модели.
    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        for i in range(self.Iterations):
            error = 0
            for Xi, target in zip(X, y):
                update = self.learn_rate * (target - self.predict(Xi))
                self.weights[1:] += update*Xi
                self.weights[0] += update
                error += int(update != 0)
            self.errors.append(error)
        return self
    
    # Метод Net Input для суммирования заданных входных данных матрицы и их соответствующих весов.
    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    # Метод Predict для прогнозирования классификации входных данных.
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_decision_regions

url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 
# url = "F:/ii ЛАБЫ/data/seeds_dataset.data" 
names = ["площадь","периметр","компактность", "длина","ширина","асимметрия","длина канавки ядра","сорт"] #название атрибутов 
target_names = { 1:"Кама" , 2:"Роза" , 3:"Канадка" }
dataset = pd.read_csv(url, names=names)
dataset.tail()

y = dataset.iloc[0:140,7].values
#тзменение классов для работы 
y = np.where(y== 2, -1, 1)
# print(y)

X = dataset.iloc[0:140, [0, 3]].values
# print(X)
plt.scatter(X[: 50, 0], X[: 50, 1], color='red' , marker='o' , label= 'Кама' ) 

plt.scatter(X[70 :140 , 0] , X[70 :140 , 1], color='blue' , marker= 'X' , label= 'Роза' ) 

plt.xlabel ( 'Площадь' ) 
plt.ylabel('длина') 
plt.legend(loc='upper left') 
plt.show() 

#
    # print(X[: 50, 0])  #Площадь первые 50 первого типа сорта КАМА
    # print(X[: 50, 1])  #длина первые 50 первого типа сорта КАМА

    # print(X[70 :120 , 0]) #Площадь с 70 по 100 строку, тут описывается РОЗА
    # print(X[70 :120 , 1]) #длина с 70 по 100 строку, тут описывается РОЗ

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

perceptron = Perceptron(Learn_Rate=0.0001, Iterations=20)
perceptron.fit(X_train, y_train)
y_predict = perceptron.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')

plt.xlabel('Эпохи')
plt.ylabel('Количество ошибочных классификаций')
plt.show()