import numpy as np
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
with open('weights.json') as json_file:
    weights = json.load(json_file)
    print(weights)

weights = {int(k):int(v) for k,v in weights.items()}


image_gen = ImageDataGenerator(rotation_range=20,
                               rescale = 1./255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')

dataset = image_gen.flow_from_directory('TrainDatasetLarge', class_mode='categorical', color_mode='rgb', batch_size=12,
                                                    target_size=(130,130), shuffle=True, seed=None, subset=None,
                                                    interpolation='bilinear', follow_links=False)

test_set = image_gen.flow_from_directory('TestDatasetLarge', class_mode='categorical', color_mode='rgb', batch_size=12,
                                                    target_size=(130,130), shuffle=True, seed=None, subset=None,
                                                    interpolation='bilinear', follow_links=False)

#test_set = tf.keras.preprocessing.image_dataset_from_directory('TestDatasetLarge', labels='inferred',
 #                                                   label_mode='categorical', color_mode='rgb', batch_size=12,
  #                                                  image_size=(150,150), shuffle=True, seed=None, validation_split=None, subset=None,
   #                                                 interpolation='bilinear', follow_links=False, smart_resize=True)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from tensorflow.keras.models import Sequential
cnn = Sequential()

K.clear_session()
image_shape = (130, 130, 3)
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout

cnn.add(Conv2D(filters=32,kernel_size=(3,3), input_shape=image_shape,activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(filters=128,kernel_size=(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=256,kernel_size=(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dropout(rate=0.2))
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dense(143,activation='softmax'))

cnn.summary()

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',patience=4)

cnn.fit(dataset, epochs=70, batch_size=2, validation_data=test_set, callbacks=[early_stopping], class_weight=weights)

cnn.save('model_15_LargeDS_largePhotos.h5')

from PIL import Image

img = Image.open('test_pok.png')
img = img.resize((130, 130))

# Dropping the transperency channel if it exists

img = np.array(img)
if img.shape[2] != 3:
    img = img[:, :, :3]

# Converting to array, normalizing and expanding the dims

img = img / 255
my_img_array = np.expand_dims(img, axis=0)

# Fething the prediction
pred_class = np.argmax(cnn.predict(my_img_array), axis=-1)
print(pred_class)
print(test_set.class_names[pred_class[0]])

