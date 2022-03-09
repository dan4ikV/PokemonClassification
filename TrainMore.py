from tensorflow import keras
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

cnn = keras.models.load_model('model_18_LargeDS_largePhotos_up.h5')


dataset = image_gen.flow_from_directory('TestDatasetLarge', class_mode='categorical', color_mode='rgb', batch_size=12,
                                                    target_size=(130,130), shuffle=True, seed=None, subset=None,
                                                    interpolation='bilinear', follow_links=False)

test_set = image_gen.flow_from_directory('TestDataset', class_mode='categorical', color_mode='rgb', batch_size=12,
                                                    target_size=(130,130), shuffle=True, seed=None, subset=None,
                                                    interpolation='bilinear', follow_links=False)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',patience=4)

cnn.fit(dataset, epochs=70, batch_size=2, validation_data=test_set, callbacks=[early_stopping], class_weight=weights)

cnn.save('model_19_LargeDS_largePhotos_up.h5')