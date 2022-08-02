import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import datetime

export_path = "c:/Users/HAMZA_MUJTABA/Documents/ElectrifyBackend/1658467060"
reloaded = tf.keras.models.load_model(export_path)

labels_path = ("c:/users/HAMZA_MUJTABA/documents/ElectrifyBackend/names.txt")
class_names = np.array(open(labels_path).read().splitlines())

resistor_path = ("c:/users/HAMZA_MUJTABA/documents/ElectrifyBackend/Test/150R_1-4W.jpg")

img = tf.keras.utils.load_img(
    resistor_path, target_size=(700, 700)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

reloaded_predicted_id = tf.math.argmax(img_array, axis=-1)
reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
print(reloaded_predicted_label_batch)

