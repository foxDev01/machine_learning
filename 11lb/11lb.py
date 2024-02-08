
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
url = "C:/Users/fox/Desktop/ii/data/seeds_dataset.data" 
names = ["площадь","периметр","компактность", "длина","ширина","асимметрия","длина канавки ядра","сорт"] #название атрибутов 
target_names = { 1:"Кама" , 2:"Роза" , 2:"Канадка" }
dataset = pd.read_csv(url, names=names)
dataset.head()
X = dataset.iloc[:,0:7].values
y = dataset['сорт'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, input_shape=(7,), name='fc1', activation='sigmoid'),
  tf.keras.layers.Dense(10, activation='softmax', name='fc2'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(8, activation='softmax', name='fc3')
])
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
print(y_test)
cl = model.fit(X_train, y_train,verbose=2, epochs=40,validation_data=(X_test,y_test),)

fig, ax = plt.subplots(figsize=(12,5))
ax.set_title('Правильность при обучении', size=15)
plt.plot(cl.history['accuracy'], label='Правильность')
ax.set_xlabel('Эпохи', size=15)
fig, ax = plt.subplots(figsize=(12,5))
ax.set_title('Потери при обучении', size=15)
plt.plot(cl.history['loss'], label='Потери')
ax.set_xlabel('Эпохи', size=15)
plt.legend()
plt.show()