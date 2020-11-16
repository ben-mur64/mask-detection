import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from keras.applications import MobileNetV2
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Activation, SeparableConv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam

batch_size = 40
img_height = 200
img_width = 200

# Best validation accuracy .9662
BUILDING = False
LOADING = True

def build_model(img_height, img_width):
    inputs = Input(shape=(img_height, img_width, 3))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "sigmoid"
    units = 1

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_ds, epochs=10, batch_size=32, validation_data=testing_ds)
    save_model(model, "./best_model")
    return model

def load():
    model = load_model('best_model')
    return model


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

if BUILDING:
    model = build_model(img_height, img_width)

if LOADING:
    model = load()

predict = model.predict(testing_ds)
predict = [1 * (x[0]>=0.5) for x in predict]
true = np.concatenate([y for x,y in testing_ds], axis=0)
print(confusion_matrix(true,predict))
print(classification_report(true,predict))
