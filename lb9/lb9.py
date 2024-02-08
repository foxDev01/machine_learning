from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis
url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 

names = ["площадь","периметр","компактность", "длина","ширина","асимметрия","длина канавки ядра","сорт"] #название атрибутов 
target_names = { 1:"Кама" , 2:"Роза" , 2:"Канадка" }
dataset = pd.read_csv(url, names=names)
dataset.head()
X = dataset.iloc[:,0:7] 
y = dataset['сорт'].values
print(X)
#---------------pca---------------------------
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
explained_variance = pca.explained_variance_ratio_
print("объясненный коэффициент дисперсии(первые два компонента):%s"%str(explained_variance))
plt.figure()
colors=['r', 'g','b']
lw=1
#------------------------построить график по первым двум главным компонентам---------
for color,i,target_name in zip(colors,[1,2,3], target_names.values()):
    plt.scatter(X_r[y==i,0], X_r[y==i,1], c=color, alpha=.8, lw=lw, label=target_name)
plt.title("PCA")
plt.legend(loc="best",shadow=False,scatterpoints=1)
plt.show()

#---------------------кумуляционная объясненная дисперсия----------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    stratify=y, random_state=0
)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print ("coбственные значения%s"% eigen_vals)
# рассчитать совокупную сумму объясненных отклонений
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# вывод графика
plt.bar(range(0,7), var_exp, alpha=0.5,
        align='center', label='индивидуальная объясненная дисперсия')
plt.step(range(0,7), cum_var_exp, where='mid',
         label='кумуляционная объясненная дисперсия')
plt.ylabel('доля объясненной дисперсии')
plt.xlabel('Главные компоненты')
plt.legend(loc='best')
plt.show()

#------------------------произвести снижение размерности призн прос-ва используя LDA---------
model = LinearDiscriminantAnalysis()
model. fit(X, y)
from sklearn. model_selection import cross_val_score
from sklearn. model_selection import RepeatedStratifiedKFold
#Define method to evaluate model
cv = RepeatedStratifiedKFold(n_splits= 10 , n_repeats= 3 , random_state= 1 )
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean (scores)) 
plt.figure()
colors = ['red', 'green', 'blue']
lw = 2
data_plot = model.fit (X, y).transform (X)
#------------------------построить график по первым двум главным компонентам---------
for color, i, target_name in zip(colors, [1, 2, 3], target_names.values()):
 plt.scatter (data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
 label=target_name)
plt.legend(loc='best', shadow= False , scatterpoints=1)
plt.title("LDA")
plt.show()

#------------------------произвести нелинейное снижение размерности призн прос-ва используя PCA и kpca---------
kpca = KernelPCA(kernel="rbf", n_components=2, gamma=.01)
z = kpca.fit_transform(X)

df = pd.DataFrame()
df["y"] = y
df["Компонент-1"] = z[:,0]
df["Компонент-2"] = z[:,1]

sns.scatterplot(x="Компонент-1", y="Компонент-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 3),
                data=df).set(title="kPCA ")
plt.show()
#------------------------t-SNE-----------------

X_tsne = TSNE(learning_rate=100, random_state=1000).fit_transform(X_r)
X_pca = PCA().fit_transform(X)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("t-SNE")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.subplot(122)
plt.title("PCA")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.show()