import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
plt.style.use("fivethirtyeight")

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Flatten, Rescaling

data_dir_train = pathlib.Path(r"..\lungxray\train")
data_dir_test = pathlib.Path(r"..\lungxray\test")
data_dir_val = pathlib.Path(r"..\lungxray\val")


print("Number of Images in Train:", len(list(data_dir_train.glob("*/*"))))
print("Number of Images in Test:", len(list(data_dir_test.glob("*"))))
print("Number of Images in Validation:", len(list(data_dir_val.glob("*/*"))))

height = 224
width = 224
batch_size = 20
seed = 42

train_dataset = keras.preprocessing.image_dataset_from_directory(data_dir_train, seed=seed, validation_split=0.2, subset='training', 
image_size=(height,width), batch_size=batch_size)

val_ds = keras.preprocessing.image_dataset_from_directory(data_dir_train, seed=seed, validation_split=0.2, subset='validation',
image_size=(height,width), batch_size=batch_size)


class_names = train_dataset.class_names

plt.figure(figsize=[10,8])

# Next we are just picking one image from the unique categories and displaying them:
for index, classes in enumerate(class_names):
    for images in train_dataset.file_paths:
        if classes in images:
            img = image.imread(images)
            plt.subplot(1,2,index+1)
            plt.imshow(img, cmap=plt.cm.gist_gray)
            plt.xticks([])
            plt.yticks([])
            plt.title(str(classes))
            break
plt.show()

model = Sequential([
    Rescaling(1./255, input_shape=(height, width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(2)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_ds, epochs=20)

pd.DataFrame(history.history).plot(figsize=[10,5])
plt.yticks(np.linspace(0,1,6))
plt.xticks(np.linspace(0,10,2))
plt.show()


# This is our trained model. We saved it so that we do not need to train it again and again.
# I am saving this in the .h5 format.

model.save("../xray_model.h5")
