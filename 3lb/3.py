import pandas as pd

url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 
names = ["площадь","периметр","компактность", "длина","ширина","асимметрия","длина канавки ядра","сорт"] #название атрибутов 
target_names = { 1:"Кама" , 2:"Роза" , 2:"Канадка" }
dataset = pd.read_csv(url, names=names)
dataset.head()
X = dataset.iloc[:,0:7] 
y = dataset['сорт'].values
print(X)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X_scal = scaler.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scal, y, test_size = 0.30)
from sklearn.neighbors import KernelDensity
kernels=["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]

for j in range(6):
    print( '\n' +(kernels[j-1]))
    kernelIndex = j-1
#выыод расчета с соответствующими kernel
    for i in range(2,12):
        kd = KernelDensity(bandwidth=i, kernel=kernels[kernelIndex])
        kd.fit(X_train)
        print("%.2f" % kd.score(X_test))
    
