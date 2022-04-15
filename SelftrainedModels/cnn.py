# CNN using VGG - 16 idea (using just conv(3x3) and max_pool(2,2) but with small number of layers.
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend
from keras import models, layers
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from data_process import load_data

X, y = load_data()
# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)

print(X_train.shape)
print(y_train.shape)

backend.clear_session()

np.random.seed(42)
tf.random.set_seed(42)

model = models.Sequential()
model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=16)

# Once the training is complete, plot the loss and accuracy metrics of the model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)
