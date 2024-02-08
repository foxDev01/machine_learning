import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

url = "C:/Users/fox/Desktop/ii/1lb/seeds_dataset.data"
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
array = dataset.values  
#( Кама , Роза , Канадка ) обозначены как числовые переменные 1, 2 и 3 соответственно. Семь исходных переменных
dataset.head()
print(array)

dataset[names[:7]].hist() 

sns.pairplot(dataset[names], height=1, hue = "Сорт") #2 диаграмма
plt.show()

corr = np.corrcoef(dataset[names[:7]].values.T)
sns.set(font_scale=1.5)
ax = sns.heatmap(
    corr,
    cbar=True,
    annot=True,
    square=True,
    linewidths=5.55,
    fmt=".2f",
    annot_kws={"size": 15},
    yticklabels=names,
    xticklabels=names,
) #3 диаграмма
plt.show()