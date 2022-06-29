# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator


def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()


    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(
        # YOUR CODE HERE
        rescale=1 / 255,
        validation_split=0.2
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator=training_datagen.flow_from_directory(
        TRAINING_DIR,
        class_mode='categorical',
        target_size=(150, 150)
    )
    validation_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        class_mode='categorical',
        target_size=(150, 150)
    )
    # YOUR CODE HERE


    model=tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(16, (3,3), input_shape=(150,150,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_generator, validation_data=validation_generator, epochs=5)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
