# For reproducible results.
# See: 
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import os, inspect
os.environ['PYTHONHASHSEED'] = '0' # Disable hash randomization

# Set current working to the parent directory of this file
#currentdir = os.get
#parentdir = os.path.dirname(currentdir)

import numpy as np
import tensorflow as tf
import random as python_random

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state with a reproducible sequence.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/guide/random_numbers
tf.random.set_seed(1234)

import itertools
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, RMSprop

import optuna

# Load data
# The MNIST dataset is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
# The dataset is available at https://s3.amazonaws.com/img-datasets/mnist.npz
# Each image is 28 pixels in height and 28 pixels in width (28 x 28), making a total of 784 pixels. Each pixel value is an integer between 0 and 255, indicating the darkness in a gray-scale of that pixel.
# The data is stored in a dataframe where each pixel is a column (so it is flattened and not in the 28 x 28 format).
# The data set the has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
data = pd.read_csv("./mnist.csv")
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('label', axis=1), 
    data['label'], 
    test_size=0.1, 
    random_state=0
    )

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Plotting with rotated x-axis labels
g = sns.countplot(x=y_train)
plt.xlabel('Digits')
plt.ylabel('Number of images')
plt.show()

# Image re-scaling
X_train = X_train / 255
X_test = X_test / 255

# Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
print("Digits:", y_train.unique())

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)
#print(y_train)

g = plt.imshow(X_train[0][:,:,0], cmap='gray')
plt.show()
g = plt.imshow(X_train[10][:,:,0], cmap='gray')
plt.show()

# 3. Define-by-Run design
path_best_model = "cnn_model.h5"
best_accuracy = 0.0

def objective(trial):
    model = Sequential()

    # add different number of convolutional layers
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    for i in range(num_conv_layers):
        # Use identical config for all layers
        model.add(Conv2D(
            filters=trial.suggest_categorical('filters', [16, 32, 64]),
            kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
            strides=trial.suggest_categorical('strides', [1, 2]),
            activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
            padding='same',
        ))
    # add pooling layer, may optimize pool_size and strides if needed
    model.add(MaxPool2D(pool_size=2, strides=2))

    # add flatten layer
    model.add(Flatten())

    # add fully connected layers
    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    for i in range(num_dense_layers):
        model.add(Dense(
            units=trial.suggest_int('units', 5, 512),
            activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
        ))

    # last fully connected layer
    model.add(Dense(10, activation='softmax'))

    # use Adam or RMSprop optimizer for training the network
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    if optimizer_name == 'Adam':
        optimizer = Adam(
            lr=trial.suggest_float('learning_rate', 1e-6, 1e-2),
        )
    else:
        optimizer = RMSprop(
            lr=trial.suggest_float('learning_rate', 1e-6, 1e-2),
            momentum=trial.suggest_float('momentum', 0.1, 0.9),
        )

    # compile model
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # train model
    history = model.fit(x = X_train, 
              y = y_train, 
              epochs=3,
              batch_size=128,
              validation_split=0.1,
              )
    
    accuracy = history.history['val_accuracy'][-1]

    # save the model if it is better than the previous best model
    global best_accuracy

    if accuracy > best_accuracy:
        model.save(path_best_model)
        best_accuracy = accuracy

    del model

    return accuracy

# prepare another objective function for the study that uses different hyperparameters for different layes of the model
def objective_2(trial):
    model = Sequential()

    # add different number of convolutional layers
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    for i in range(num_conv_layers):
        # Use different config for each layer
        model.add(Conv2D(
            filters=trial.suggest_categorical('filters_{}'.format(i), [16, 32, 64]),
            kernel_size=trial.suggest_categorical('kernel_size_{}'.format(i), [3, 5]),
            strides=trial.suggest_categorical('strides_{}'.format(i), [1, 2]),
            activation=trial.suggest_categorical('activation_{}'.format(i), ['relu', 'tanh']),
            padding='same',
        ))
    # add pooling layer, may optimize pool_size and strides if needed
    model.add(MaxPool2D(pool_size=2, strides=2))

    # add flatten layer
    model.add(Flatten())

    # add fully connected layers
    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    for i in range(num_dense_layers):
        model.add(Dense(
            units=trial.suggest_int('units{}'.format(i), 5, 512),
            activation=trial.suggest_categorical('activation{}'.format(i), ['relu', 'tanh']),
        ))

    # last fully connected layer
    model.add(Dense(10, activation='softmax'))

    # use Adam or RMSprop optimizer for training the network
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    if optimizer_name == 'Adam':
        optimizer = Adam(
            lr=trial.suggest_float('learning_rate', 1e-6, 1e-2),
        )
    else:
        optimizer = RMSprop(
            lr=trial.suggest_float('learning_rate', 1e-6, 1e-2),
            momentum=trial.suggest_float('momentum', 0.1, 0.9),
        )

    # compile model
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # train model
    history = model.fit(x = X_train, 
              y = y_train, 
              epochs=3,
              batch_size=128,
              validation_split=0.1,
              )
    
    accuracy = history.history['val_accuracy'][-1]

    # save the model if it is better than the previous best model
    global best_accuracy

    if accuracy > best_accuracy:
        model.save(path_best_model)
        best_accuracy = accuracy

    del model

    return accuracy

study_name = 'cnn_study'
storage_name = 'sqlite:///{}.db'.format(study_name)

study = optuna.create_study(
    direction='maximize',
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

study.optimize(objective, n_trials=5)

# 4. Analyze the results
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")

results = study.trials_dataframe()
results['value'].sort_values().reset_index(drop=True).plot()
plot_title = 'Convergence plot'
plt.xlabel('Trial')
plt.ylabel('Validation Accuracy')
plt.show()

# 5. Evaluate the model
model = load_model(path_best_model)
print(model.summary())

result = model.evaluate(x=X_test, y=y_test)

# print evaluation metrics
for name, value in zip(model.metrics_names, result):
    print(name, value)

# 5.1 Confision matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

confusion_mtx = confusion_matrix(y_true, y_pred_classes)
print(confusion_mtx)

classes = 10
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(classes)
plt.xticks(tick_marks, range(classes), rotation=45)
plt.yticks(tick_marks, range(classes))
for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    plt.text(j, i, confusion_mtx[i, j], 
             horizontalalignment='center', 
             color='white' if confusion_mtx[i, j] > 100 else 'black')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()