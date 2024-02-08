from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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
X = X.values
# Y = dataset['сорт']
# Y = Y.values
print(X)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
Y_km= km.fit_predict(X) 
plt.scatter(X[Y_km==0,0], X[Y_km==0,1], s=50, c='lightgreen', marker='s', label='кластер 1') #бере все Х с 0 индексом где У соответ = 0 
plt.scatter(X[Y_km==1,0], X[Y_km==1,1], s=50, c='orange',marker='o', label='кластер 2')
plt.scatter(X[Y_km==2,0], X[Y_km==2,1], s=50,  c='lightblue', marker='v', label='кластер 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250,  c='red', marker='*')

plt.legend()
plt.grid()
plt.show()
print("Искажения: %.2f" % km.inertia_)
distortions=[]

for i in range(1,11):
    km=KMeans(n_clusters=i,
              init='k-means++',
              n_init=10,
              max_iter=300,
              random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11),distortions,marker='o')
plt.xlabel("Количество кластеров")
plt.ylabel("Искажения ")
plt.grid()
plt.savefig("./fig3.png")
plt.show()
km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = [] 
for i , с in enumerate(cluster_labels):
  c_silhouette_vals = silhouette_vals[y_km == с]
  c_silhouette_vals.sort()
  y_ax_upper += len(c_silhouette_vals)
  color = cm.jet(float(i) / n_clusters)
  plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor = 'none', color=color)
  yticks.append((y_ax_lower + y_ax_upper) / 2)
  y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals) 
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)

plt.ylabel('Кластер') 
plt.xlabel('Силуэтный коэффициент')
plt.show () 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dbscan = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
y_db = dbscan.fit_predict(X)
 
# Готово! Распечатаем метки принадлежности к кластерам
print(dbscan.labels_)
 
# Строим в соответствии с тремя классами
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)

for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')



# plt.scatter(X[y_db==0,-1], X[y_db==0,1], s=40, c='lightgreen', marker='s', label='кластер 1')
# plt.scatter(X[y_db==1,0], X[y_db==1,1], s=40, c='red', marker='o', label='кластер 2')
# plt.scatter(X[y_db==2,0], X[y_db==2,1], s=40, c='blue', marker='v', label='кластер 3')

plt.legend()
plt.show()

# https://proglib.io/p/unsupervised-ml-with-python

from sklearn.cluster import OPTICS

clustering = OPTICS(min_samples=20).fit(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print (clustering.labels_)



X['PC1'] = pca.fit_transform(X)[:,0]
X['PC2'] = pca.fit_transform(X)[:,1]
X['clustering'] = clustering.labels_

sns.scatterplot(data=X,x="PC1",y="PC2",hue=X['clustering'])