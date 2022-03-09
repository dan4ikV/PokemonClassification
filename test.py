import numpy as np
import tensorflow as tf

cnn = tf.keras.models.load_model('model_19_LargeDS_largePhotos_up.h5')

test_set = tf.keras.preprocessing.image_dataset_from_directory('TestDataset', labels='inferred',
                                                    label_mode='categorical', color_mode='rgb', batch_size=12,
                                                    image_size=(200,200), shuffle=True, seed=None, validation_split=None, subset=None,
                                                    interpolation='bilinear', follow_links=False, smart_resize=True)
from PIL import Image

img = Image.open('bulbasaur_fanart.jpg')
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