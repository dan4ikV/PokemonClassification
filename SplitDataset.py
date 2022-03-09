import numpy as np
import os
import tensorflow as tf
import random
from shutil import copyfile
import json

tf.keras.backend.clear_session()

labels = np.array(os.listdir('PokemonDatasetLarge'))
labels
weights = {}

for i, pokemon in enumerate(labels):
    images = os.listdir(f'PokemonDatasetLarge/{pokemon}')
    images = [image for image in images if image.endswith('.png') or image.endswith('.jpg')]
    random.shuffle(images)
    weights[i] = len(images)
    train_images = images[:int((len(images) * 0.7))]
    os.makedirs(f'TrainDatasetLarge/{pokemon}')
    os.makedirs(f'TestDatasetLarge/{pokemon}')
    test_images = images[int(len(images) * 0.7):]
    for image in train_images:
        copyfile(f'PokemonDatasetLarge/{pokemon}/{image}', f'TrainDatasetLarge/{pokemon}/{image}')
    for image in test_images:
        copyfile(f'PokemonDatasetLarge/{pokemon}/{image}', f'TestDatasetLarge/{pokemon}/{image}')

weights = json.dumps(weights)
jsonFile = open("weights.json", "w")
jsonFile.write(weights)
jsonFile.close()