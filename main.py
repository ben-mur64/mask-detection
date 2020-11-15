import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam

batch_size = 40
img_height = 200
img_width = 200


# Keras dataset preprocessing utilities, located at tf.keras.preprocessing,
# help you go from raw data on disk to a tf.data.Dataset object that can be used to train a model.
# 
# See https://keras.io/api/preprocessing/
# 
# Once the dataset is preprocessed and loaded, it can be directly used in calls to model.fit
# 
# 
## loading training data
training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/',
    validation_split=0.2,
    subset= "training",
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size

)

## loading testing data
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
'data/',
    validation_split=0.2,
    subset= "validation",
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size

)

class_names = training_ds.class_names

# plt.figure(figsize=(10, 10))
# for images, labels in training_ds.take(1):
#   for i in range(12):
#     ax = plt.subplot(3, 4, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.grid(True)
# plt.show()

## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)


# # Now build a deep neural network and train it and see how you do

# 
inputs = Input(shape=(img_height, img_width, 3))

# 2 Convolutional layers each followed by max pooling
x = Conv2D(200, kernel_size=(5, 5), kernel_initializer='he_normal')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(100, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Here we'll add a single Dense layer before the prediction
x = Flatten()(x)
x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(training_ds, epochs=5, batch_size=32, validation_data=testing_ds)
