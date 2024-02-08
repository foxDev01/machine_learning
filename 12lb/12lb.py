import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
class_names = ['cat','chicken','dog']
path = 'C:/Users/fox/Desktop/ii/12lb/my_image'
os.listdir(path)
train_dataset = image_dataset_from_directory(
    path,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    image_size = (150,150),
    batch_size = 256,
     validation_split=0.1,
    seed=42,
    subset = 'training'
    )
validation_dataset = image_dataset_from_directory(
    path,
    image_size = (150,150),
    validation_split=0.1,
    batch_size = 256,
    seed=42,
    subset = 'validation'
    )
test_dataset = image_dataset_from_directory(
    path,
    image_size = (150,150),
    batch_size = 256,
    )
# Настраиваем производительность TensorFlow DataSet'ов
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Создаем последовательную модель
model = tf.keras.models.Sequential([
    # Сверточный слой
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape = (150,150,3),padding='same'),
    # Слой подвыборки
    tf.keras.layers.MaxPooling2D(2,2),
    # Сверточный слой
    tf.keras.layers.Conv2D(64,(3,3), activation='relu',padding='same'),
    # Слой подвыборки
    # tf.keras.layers.MaxPooling2D(2,2),
    # # Сверточный слой
    # tf.keras.layers.Conv2D(128,(3,3), activation='relu',padding='same'),
    # Слой подвыборки
    tf.keras.layers.MaxPooling2D(2,2),
    # Полносвязная часть нейронной сети для классификации
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # Выходной слой, 3 нейрона по количеству классов
    tf.keras.layers.Dense(3, activation='softmax')
    ])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
history = model.fit(train_dataset, 
                    validation_data=validation_dataset,
                    epochs=10,
                    verbose=2)
# Оцениваем качество обучения модели на тестовых данных
scores = model.evaluate(test_dataset, verbose=1)
plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 
         label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'], 
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

# class_names = ['cat','chicken','dog']

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
model.save("cat_chicken_dog_model.h5")