import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y= y.astype(int)
# X=((X/225.)-.5*2)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=123, test_size=100, stratify=y) 
print(y_test)
print('strok:%d, stolbcov:%d'%(X_train.shape[0],X_train.shape[1]))
print('strok:%d, stolbcov:%d'%(X_test.shape[0],X_test.shape[1]))

for i in range(25):
    digit =X.loc[i*2500].to_numpy()
    digit_image = digit.reshape(28,28)
    plt.subplot(5,5, i +1)
    plt.axis('off')
    plt.imshow(digit_image, cmap='gray')
plt.show()
