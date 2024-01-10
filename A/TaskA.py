# Import Python Library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score

# Set to give consistent results
tf.keras.utils.set_random_seed(0)

# Define file path
path = r"Dataset\pneumoniamnist.npz"

# Unpack numpy dataset file
with np.load(path) as data:
    X_train = data['train_images']
    y_train = data['train_labels']
    X_test = data['test_images']
    y_test = data['test_labels']
    X_val = data['val_images']
    y_val = data['val_labels']
    
# Reshape numpy array and datatype
X_train = X_train.reshape([-1, 28, 28, 1]).astype('float32')
X_val = X_val.reshape([-1, 28, 28, 1]).astype('float32')
X_test = X_test.reshape([-1, 28, 28, 1]).astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_val /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 2
Y_train = tf.keras.utils.to_categorical(y_train, n_classes)
Y_val = tf.keras.utils.to_categorical(y_val, n_classes)
Y_test = tf.keras.utils.to_categorical(y_test, n_classes)

# Define CNN model
model = tf.keras.Sequential()
# Input layer
model.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))
# Convolutional layer
model.add(tf.keras.layers.Conv2D(16, kernel_size=(5,5), strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = 2))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=1, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = 2))
# Fully connected dense layer
# flatten output of conv
model.add(tf.keras.layers.Flatten())
# hidden layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
# output layer
model.add(tf.keras.layers.Dense(2, activation='softmax'))
# compile CNN model with ADAM optimizer
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam())
# Train model
cnn_model = model.fit(X_train,Y_train, epochs = 150, validation_data = (X_val,Y_val))