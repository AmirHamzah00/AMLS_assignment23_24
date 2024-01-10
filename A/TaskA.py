# Python Library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
# Set to make results consistent
tf.keras.utils.set_random_seed(0)
# Define File Path
path = r"Dataset\pneumoniamnist.npz"
# Unpack Numpy Dataset File
with np.load(path) as data:
    X_train = data['train_images']
    y_train = data['train_labels']
    X_test = data['test_images']
    y_test = data['test_labels']
    X_val = data['val_images']
    y_val = data['val_labels']
# Reshape Numpy array and define datatype
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

# function to create, train and evaluate CNN model
def Train_Evaluate_CNN_Model_TaskA():
    # Define CNN model
    model = tf.keras.Sequential()
    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))
    # Convolution layer
    # 1st layer
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(5,5), strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization()) # added batch normalization
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = 2))
    model.add(tf.keras.layers.Dropout(0.2)) # added dropout layer
    # 2nd layer
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=1, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization()) # added batch normalization
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = 2))
    model.add(tf.keras.layers.Dropout(0.2)) # added dropout layer
    # Dense layer
    # flatten output of conv
    model.add(tf.keras.layers.Flatten())
    # hidden layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2)) # added dropout layer
    # output layer
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # Define learning rate schedule
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.01)
    # created callback array 
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)]
    # compile CNN model with ADAM optimizer and callbacks
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5))
    # train model
    cnn_model = model.fit(X_train,Y_train, batch_size=64, epochs = 150, callbacks=callback, validation_data = (X_val,Y_val))
    
    # Predict test dataset with trained CNN model
    y_pred = model.predict(X_test)
    # Store predicted class in list
    predicted = []
    expected = y_test
    for x in range(len(y_pred)):
        prediction_set = y_pred[x]
        max_index = np.argmax(prediction_set)
        if max_index == 0:
            predicted.append(0)
        else:
            predicted.append(1)
    # calculating the f-score
    fscore = f1_score(expected, predicted)
    # print f-score and accuracy score
    print('the f-score is: %.4f' % (fscore))
    print('the accuracy is: %.2f' % (accuracy_score(expected, predicted)*100) + '%')
    # Confusion Matrix created for evaluation
    cm = tf.math.confusion_matrix(expected, predicted)
    # Plot Results for analysis
    # Accuracy train and validate graph
    plt.figure(1)
    plt.plot(cnn_model.history['accuracy'])
    plt.plot(cnn_model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    # loss train and validate graph
    plt.figure(2)
    plt.plot(cnn_model.history['loss'])
    plt.plot(cnn_model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    # plot confusion matrix
    plt.figure(3)
    plt.title('Confusion Matrix for Task A')
    heatmap = sns.heatmap(cm, annot=True, cmap="Oranges")
    heatmap.set(xlabel='Predicted', ylabel='Expected')
    # show all plots
    plt.show()

def Load_Trained_CNN_Model_TaskA(model_id):
    # load the saved CNN Model based on ID
    if model_id == 1:
        model = tf.keras.models.load_model('A\Saved_Models_TaskA\Model_TaskA_1.keras')
    elif model_id == 2:
        model = tf.keras.models.load_model('A\Saved_Models_TaskA\Model_TaskA_2.keras')
    elif model_id == 3:
        model = tf.keras.models.load_model('A\Saved_Models_TaskA\Model_TaskA_3.keras')
    else:
        raise TypeError("Invalid Model ID!") # if ID invalid error is raised
    # Predict test dataset with trained CNN model
    y_pred = model.predict(X_test)
    # predicted class is store in the list
    predicted = []
    expected = y_test
    for x in range(len(y_pred)):
        prediction_set = y_pred[x]
        max_index = np.argmax(prediction_set)
        if max_index == 0:
            predicted.append(0)
        elif max_index == 1:
            predicted.append(1)
        else:
            raise TypeError("Incorrect Classification Index Found!")
    # f-score and accuracy calc and display
    fscore = f1_score(expected, predicted)
    print('the f-score is: %.4f' % (fscore))
    print('the accuracy is: %.2f' % (accuracy_score(expected, predicted)*100) + '%')

    # Making the Confusion Matrix
    cm = tf.math.confusion_matrix(expected, predicted)
    # plot confusion matrix
    plt.title('Confusion Matrix for Task A')
    heatmap = sns.heatmap(cm, annot=True, cmap="Oranges")
    heatmap.set(xlabel='Predicted', ylabel='Expected')
    
    plt.show()
            


