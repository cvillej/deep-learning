# https://www.tensorflow.org/tutorials/keras/basic_classification
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

checkpoint_path = "/Users/juser/dev/deep-learning/models/fashion/fashion.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


def create_model():

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model):
    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])


def evaluate_model(model):
    model.load_weights(checkpoint_path)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    print('all pred 0: {}'.format(predictions[0]))
    print('hightest pred: {}'.format(np.argmax(predictions[0])))
    print('pred class name: {}'.format(class_names[test_labels[0]]))

    # predict single image
    # Grab an image from the test dataset
    img = test_images[0]
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))
    predictions_single = model.predict(img)

model = create_model()
# train_model(model)

model = create_model()
evaluate_model(model)

