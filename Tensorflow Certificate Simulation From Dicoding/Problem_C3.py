# =======================================================================================================
# PROBLEM C3
#
# Build a CNN based classifier for Cats vs Dogs dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is originally published in https://www.kaggle.com/c/dogs-vs-cats/data
#
# Desired accuracy and validation_accuracy > 72%
# ========================================================================================================

import tensorflow as tf
import urllib.request
import zipfile
import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def solution_C3():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/cats_and_dogs.zip'
    urllib.request.urlretrieve(data_url, 'cats_and_dogs.zip')
    local_file = 'cats_and_dogs.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    BASE_DIR = 'data/cats_and_dogs_filtered'
    train_dir = os.path.join(BASE_DIR, 'train')
    validation_dir = os.path.join(BASE_DIR, 'validation')

    # YOUR CODE HERE
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #rotation_range=40,
        #vertical_flip=True,
        #fill_mode='nearest',
        #zoom_range=0.2
    )
    val_datagen = ImageDataGenerator(
        rescale=1 / 255,
    )
    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        class_mode='categorical',
        #batch_size=64,
        target_size=(150, 150)
    )
    val_generator = val_datagen.flow_from_directory(
        validation_dir,
        class_mode='categorical',
        #batch_size=64,
        target_size=(150, 150)
    )
    # YOUR CODE HERE

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by 'sigmoid'
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=(150, 150, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        #tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    class myCallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epochs, logs={}):
            if(logs.get('acc') > 0.72 and logs.get('val_acc') > 0.72):
                self.model.stop_training = True

    callback = myCallbacks()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=callback)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C3()
    model.save("model_C3.h5")
