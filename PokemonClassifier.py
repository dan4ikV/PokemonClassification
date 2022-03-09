import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import multiprocessing
from keras import backend as K
from shutil import copyfile

tf.keras.backend.clear_session()

labels = np.array(os.listdir('PokemonDataset'))
labels

X_train = []
X_test = []
y_train = []
y_test = []

from PIL import Image
import random

for pokemon in labels:
    images = os.listdir(f'PokemonDataset/{pokemon}')
    images = [image for image in images if image.endswith('.png') or image.endswith('.jpg')]
    random.shuffle(images)
    train_images = images[:int((len(images) * 0.7))]
    os.makedirs(f'TrainDataset/{pokemon}')
    os.makedirs(f'TestDataset/{pokemon}')
    test_images = images[int(len(images) * 0.7):]
    for image in train_images:
        copyfile(f'PokemonDataset/{pokemon}/{image}', f'TrainDataset/{pokemon}/{image}')
        #X_train.append(image)
        #y_train.append(pokemon)
    for image in test_images:
        copyfile(f'PokemonDataset/{pokemon}/{image}', f'TestDataset/{pokemon}/{image}')
        #X_test.append(image)
        #y_test.append(pokemon)

for i, file in enumerate(X_train):
    folder = y_train[i]
    img = Image.open(f'PokemonDataset/{folder}/{file}').convert('P')
    img = img.resize((120, 120))
    test_img = img_to_array(img)
    X_train[i] = test_img

for i, file in enumerate(X_test):
    folder = y_test[i]
    img = Image.open(f'PokemonDataset/{folder}/{file}').convert('P')
    img = img.resize((120, 120))
    test_img = img_to_array(img)
    X_test[i] = test_img

X_test = np.array(X_test) / 255
X_train = np.array(X_train) / 255
y_test = np.array(y_test)
y_train = np.array(y_train)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from tensorflow.keras.models import Sequential
cnn = Sequential()

K.clear_session()

from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D

cnn.add(Conv2D(filters=64, activation='relu', input_shape=(120, 120, 1), kernel_size=(5, 5)))

cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(filters=128, activation='relu', kernel_size=(5, 5)))

cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(filters=256, activation='relu', kernel_size=(5, 5)))

cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(filters=512, activation='relu', kernel_size=(5, 5)))

cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Flatten())

cnn.add(Dense(units=2000, activation='relu'))

cnn.add(Dense(units=150, activation='softmax'))

cnn.summary()

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

y_train = one_hot(y_train)
y_train

y_test = one_hot(y_test)
y_test


def shuffle_two(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

X_train, y_train = shuffle_two(X_train, y_train)
X_test, y_test = shuffle_two(X_test, y_test)

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=150, batch_size=5, validation_split=0.1)

predicted = cnn.predict(X_test)

correct = 0
for pred, exp in zip(predicted, y_test):
    if np.argmax(pred) == np.argmax(exp):
        correct += 1

print(f'Precision: {correct / len(y_test)}')
