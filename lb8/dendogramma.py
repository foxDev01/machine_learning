from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
import seaborn as sns


url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 

names = [
    "площадь",                 
    "периметр",             
    "компактность", 
    "длина",         
    "ширина",          
    "асимметрия",
    "длина канавки ядра",        
    "сорт"            
] #название атрибутов 

dataset = pd.read_csv(url, names=names) #чтение файла с данными 
dataset.head()
X = dataset[[
    "площадь",                 
    "периметр",             
    "компактность", 
    "длина",         
    "ширина",          
    "асимметрия",
    "длина канавки ядра",                    
]]

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage 

from sklearn.cluster import AgglomerativeClustering

print(X)
cl = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
labels = cl.fit_predict(X)
print('Meтки кластеров: %s ' % labels) 

X = X.values

plt.scatter(x=X[:,0], y=X[:,1], c= cl.labels_, cmap='rainbow' )
plt.show()


row_clusters = linkage(X, method='complete', metric='euclidean')
# print(row_clusters)

link_df = pd.DataFrame(row_clusters, index=[f'step{i+1}' for i in range(row_clusters.shape[0])], 
                        columns = ['кластер1','кластер2','кластер3','кластер4' ])

row_dendr = dendrogram(link_df) 
link_df.head()
plt.tight_layout() 
plt.show()


