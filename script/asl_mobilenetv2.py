# -*- coding: utf-8 -*-
"""ASL_MobileNetV2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xsunX7Qj_XWBZwcZLyjsKBg4RI0DNo2-

# American Sign Language - MobileNetV2
Author: [Sayan Nath](https://github.com/sayannath)

Dataset Link: [Kaggle ASL](https://www.kaggle.com/grassknoted/asl-alphabet)

## Initial Setup
"""

!nvidia-smi

!pip install -q kaggle
!pip install -qq tensorflow-addons

from google.colab import files
files.upload()

"""## Data Gathering"""

!mkdir ~p ~/.kaggle
!cp kaggle.json ~/.kaggle/

#Change the permission
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d grassknoted/asl-alphabet
!unzip -q asl-alphabet.zip

mkdir train

import shutil
shutil.make_archive('dataset', 'zip', '/content/asl_alphabet_train/asl_alphabet_train')

!pip install patool

import patoolib
patoolib.extract_archive("dataset.zip", outdir="train/")

rm -rf asl_alphabet_train

"""## Setting up Path"""

train_dir = 'train/'

"""## Import the modules"""

import tensorflow as tf
tf.random.set_seed(42)

print(tf.__version__)

from imutils import paths
from pprint import pprint
from collections import Counter
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import re 

import os
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns

import numpy as np
np.random.seed(42)

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

"""## Determine the dimension of the images"""

dim1 = []
dim2 = []
for image_filename in os.listdir(train_dir+'A'):
    
    img = mpimg.imread(train_dir+'A'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(dim1,dim2)

print(np.mean(dim1))
print(np.mean(dim2))

"""`Height` is 200 and `Width` is 200"""

image_paths = list(paths.list_images("train"))
np.random.shuffle(image_paths)
image_paths[:5]

"""## Counting number of images per class"""

labels = []
for image_path in image_paths:
    label = image_path.split("/")[1]
    labels.append(label)
class_count = Counter(labels) 
pprint(class_count)

"""Wow! Balanced Dataset

## Define the Hyperparamteres
"""

TRAIN_SPLIT = 0.9
BATCH_SIZE = 256
AUTO = tf.data.AUTOTUNE
EPOCHS = 100
IMG_SIZE = 224
NUM_CLASSES=29

"""## Splitting the dataset"""

i = int(len(image_paths) * TRAIN_SPLIT)

train_paths = image_paths[:i]
train_labels = labels[:i]
validation_paths = image_paths[i:]
validation_labels = labels[i:]

print(len(train_paths), len(validation_paths))

"""## Labelling the dataset"""

le = LabelEncoder()
train_labels_le = le.fit_transform(train_labels)
validation_labels_le = le.transform(validation_labels)
print(train_labels_le[:5])

"""## Preprocessing the data"""

@tf.function
def load_images(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return (image, label)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

"""## Creating the `Data` Pipeline"""

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_le))

train_ds = (
    train_ds
    .shuffle(1024)
    .map(load_images, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

val_ds = tf.data.Dataset.from_tensor_slices((validation_paths, validation_labels_le))
val_ds = (
    val_ds
    .map(load_images, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

"""## Define the Model"""

def get_training_model(trainable=False):
    # Load the MobileNetV2 model but exclude the classification layers
    EXTRACTOR = MobileNetV2(weights="imagenet", include_top=False,
                    input_shape=(224, 224, 3))
    # We will set it to both True and False
    EXTRACTOR.trainable = trainable
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = GlobalAveragePooling2D()(class_head)
    class_head = Dense(512, activation="relu")(class_head)
    class_head = Dropout(0.5)(class_head)
    class_head = Dense(NUM_CLASSES, activation="softmax")(class_head)

    # Create the new model
    classifier = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    # Compile and return the model
    classifier.compile(loss="sparse_categorical_crossentropy", 
                          optimizer="adam",
                          metrics=["accuracy"])

    return classifier

"""## Plot"""

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Training Progress")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epochs")
    plt.legend(["train_accuracy", "val_accuracy", "train_loss", "val_loss"], loc="upper left")
    plt.show()

"""## Define the Callback"""

train_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=2, restore_best_weights=True)
]

"""## Train the Model"""

classifier = get_training_model()
h = classifier.fit(train_ds,
               validation_data=val_ds,
               epochs=EPOCHS,
               batch_size=BATCH_SIZE,
               callbacks=train_callbacks)

accuracy = classifier.evaluate(val_ds)[1] * 100
print("Accuracy: {:.2f}%".format(accuracy))

plot_hist(h)

"""## Saving our model"""

classifier.save('asl_model')

!du -lh asl_model

"""## Saving the h5 file"""

KERAS_ASL_FILE = 'asl.h5'
classifier.save(KERAS_ASL_FILE)

"""## Helper Function - To determine the file size of our model"""

import os
from sys import getsizeof

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File Size: ' + str(round(size/1024, 3)) + 'Kilobytes')
    elif unit == 'MB':
        return print('File Size: ' + str(round(size/(1024*1024), 3)) + 'Megabytes')
    else:
        return print('File Size: ' + str(size) + 'bytes')

convert_bytes(get_file_size(KERAS_ASL_FILE), "MB")

"""Wow! 16MB"""

converter = tf.lite.TFLiteConverter.from_saved_model("asl_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("asl_optimise.tflite", 'wb').write(tflite_model)
print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0))

"""## Zipping our model together"""

!tar cvf asl_model.tar.gz asl_model asl.h5 asl.tflite asl_optimise.tflite

"""## Testing Pipeline"""

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

test_image_paths = (list(paths.list_images("test")))
print(f"Total test images: {len(test_image_paths)}")

test_image_paths[:28]

test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = (
    test_ds
    .map(preprocess_image)
    .batch(BATCH_SIZE)
)

test_predictions = np.argmax(classifier.predict(test_ds), 1)

test_predictions.shape

test_predictions[:28]

test_predictions_le = le.inverse_transform(test_predictions)

test_predictions_le[:28]

interpreter = tf.lite.Interpreter(model_path = 'asl.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

interpreter.resize_tensor_input(input_details[0]['index'], (28, 224, 224, 3))
interpreter.resize_tensor_input(output_details[0]['index'], (28, 29))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

test_imgs = next(iter(test_ds))

interpreter.set_tensor(input_details[0]['index'], test_imgs)
interpreter.invoke()

tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])

print("Prediction results shape:", tflite_model_predictions.shape)
prediction_classes = np.argmax(tflite_model_predictions, axis=1)

prediction_classes[:28]

prediction_classes_le = le.inverse_transform(prediction_classes)
prediction_classes_le[:28]

