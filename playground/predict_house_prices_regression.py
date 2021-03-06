from __future__ import print_function
import os
import shutil
import tensorflow as tf
from tensorflow import keras

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import pandas as pd

import matplotlib.pyplot as plt

boston_housing = keras.datasets.boston_housing

cache_path = '/root/.keras/datasets'
if os.path.exists(cache_path) == False:
  shutil.copytree('./datasets', cache_path )

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))

train_data = train_data[order]
train_labels = train_labels[order]

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
            input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['mae'])

    return model

model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

history = model.fit(train_data, train_labels, epochs=EPOCHS,
    validation_split=0.2, verbose=0, callbacks=[PrintDot()])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
        label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
        label='Val loss')
    plt.legend()
    plt.ylim([0, 5])

plot_history(history)
plt.savefig('img1.png')

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
    validation_split=0.2, verbose=0,
    callbacks=[early_stop, PrintDot()])

plot_history(history)
plt.savefig('img2.png')

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

print(test_predictions)