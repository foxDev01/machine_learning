from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 
names = ["площадь","периметр","компактность", "длина","ширина","асимметрия","длина канавки ядра","сорт"] #название атрибутов 
target_names = { 1:"Кама" , 2:"Роза" , 2:"Канадка" }
dataset = pd.read_csv(url, names=names)
X = dataset[["площадь"]].values 
y = dataset[["компактность"]].values 
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 7].values
dataset.head()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.40,  random_state = 1)
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
regressor = LinearRegression(  )
# обучить модель с помощью обучающих наборов
regressor.fit(X_train, y_train)
# прогонозы на тестовом наборе
y_pred = regressor.predict(X_test)
# визуализация
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('LinearRegression')
plt.xlabel('площадь')
plt.ylabel('компактность')
plt.show()
# -------------RANSAC------------------
from sklearn.linear_model import RANSACRegressor 
ransac = RANSACRegressor() 
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(9, 23, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='He-выбросы')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='lightgreen', marker='s', label='Выбросы')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel("площадь")
plt.ylabel('компактность')
plt.legend(loc='upper left')
plt.show()
# -------------оценивание------------------
# -------------остаточный график------------------
X = dataset.iloc[:, :-2].values
y = dataset['длина канавки ядра'].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.40,  random_state = 1)
slr = LinearRegression () 
slr.fit (X_train, y_train) 
y_train_pred = slr.predict(X_train) 
y_test_pred = slr.predict(X_test) 
plt.scatter(y_train_pred, y_train_pred - y_train, 
            c='blue', marker= 'o', label='Тренировочны еданные' ) 
plt .scatter (y_test_pred, y_test_pred - y_test, 
            c= 'lightgreen' , marker= 's',label= 'Тестовые данные ') 
plt.xlabel('Пр едсказанные значения') 
plt.ylabel('Остатoк' ) 
plt.legend(loc='upper left') 
plt.hlines(y=0, xmin=-10 , xmax=50 , lw=2, color= 'red') 
plt.xlim([- 10 , 50 ]) 
plt.show() 
#--------------------среднеквадр
from sklearn.metrics import mean_squared_error 
print('MSE тренировка:', mean_squared_error(y_train, y_train_pred)) 
print('MSE тестирвание:', mean_squared_error(y_test , y_test_pred))
# Мы увидим, что MSE на тренировочном наборе равна 0.02127322889789492, а MSE тестового набора намного больше со значением 0.018740387284111308, 
# что указывает на то, что наша модель переподогнана под тренировочные данные. 
from sklearn.metrics import r2_score
print('R 2 тренировка:', r2_score(y_train, y_train_pred)) 
print('R 2 тестирвание:', r2_score(y_test , y_test_pred))
# После оценивания на тренировочных данных коэффициент детерминации R2 нашей модели состав яет 0.911107122311821, что является неплохим результатом. 
# Однако R2 на тестовом наборе данных составил всего 0.9208298600711757
#-------------------применение регуляризованых методов 
X = dataset[["площадь"]].values 
y = dataset[["компактность"]].values 

dataset.head()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.40,  random_state = 1)
#--------------------гребнева 
from sklearn.linear_model import Ridge 
ridge = Ridge(alpha=1.0)
# обучить модель с помощью обучающих наборов
ridge.fit(X_train, y_train)
# прогонозы на тестовом наборе
y_pred = ridge.predict(X_test)
# визуализация
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, ridge.predict(X_train), color = 'blue')
plt.title('Ridge')
plt.xlabel('площадь')
plt.ylabel('компактность')
plt.show()
#--------------------lasso 
from sklearn.linear_model import Lasso 
lasso = Lasso(alpha=1.0) 
# обучить модель с помощью обучающих наборов
lasso.fit(X_train, y_train)
# прогонозы на тестовом наборе
y_pred = lasso.predict(X_test)
# визуализация
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lasso.predict(X_train), color = 'blue')
plt.title('lasso')
plt.xlabel('площадь')
plt.ylabel('компактность')
plt.show()
#--------------------эластичная сеть
from sklearn.linear_model import ElasticNet 
lassoE = ElasticNet(alpha=1.0, l1_ratio=0.5) 
# обучить модель с помощью обучающих наборов
lassoE.fit(X_train, y_train)
# прогонозы на тестовом наборе
y_pred = lassoE.predict(X_test)
# визуализация
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lassoE.predict(X_train), color = 'blue')
plt.title('lassoE')
plt.xlabel('площадь')
plt.ylabel('компактность')
plt.show()
#--------------------полиномиальная 
from sklearn.preprocessing import PolynomialFeatures
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
# Для равнения выполнить подгонку простой линейной регрессионной модели :
lr.fit(X, y)
X_fit = np.arange(9, 23, 1) [:, np.newaxis]
y_lin_fit = lr.predict(X_fit)
# полнить подгонку множественной регрессионной модели на преобразованных при наках я полиномиальной регр ссии: 
pr.fit (X_quad, y) 
y_quad_fit = pr.predict (quadratic.fit_transform(X_fit )) 
plt.scatter(X, y, label='тренировочные точки')
plt.plot (X_fit, y_lin_fit,
            label='линейная подгонка', linestyle='--')
plt.plot (X_fit, y_quad_fit,
            label='квадратичная подгонка')
plt.legend(loc='upper left')
plt.show()
y_lin_pred = lr.predict (X)
y_quad_pred = pr.predict (X_quad)
print('Тренировочная MSE линейная:', (mean_squared_error(y, y_lin_pred)))
print('Тренировочная MSE квадратичная:', (mean_squared_error(y, y_quad_pred)))
print('Тренировочная R*2 линейная:', (r2_score(y, y_lin_pred)))                                                      
print('Тренировочная R*2 квадратичная:', (r2_score(y, y_quad_pred)))
# Как видно после выполнения приведенного выше исходного кода, средневзвешенная 
# квадатичная ошибка (MSE) в этой отдельно взятой миниатюрной задаче 
# уменьшилась с 0.0003500768067586305 (линейная подгонка) до 0.000288080960033762 (квадратичная подгонка) , при этом 
# коэффициент детерминации отражает более тесную подгонку к квадратичной модели 
# (R2 = .481580216418252) в противоположность линейной подгонке (R2 = 0.3700147959256693). 
