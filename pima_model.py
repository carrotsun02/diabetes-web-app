# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import os
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# read the file containing the pima indians diabities data set
data = pd.read_csv('./diabetes.csv', sep=',')

print("\ndata.head(): \n",
      data.head())

# describe the columns of the data set
data.describe()

# see if the data set has null values
data.info()

print("\n\nStep 2 - Prepare the data for the model building")
# extract the X and y from the imported data
X = data.values[:,0:8]
y = data.values[:,8]

# use MinMaxScaler to fit a scaler object
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the test set into the train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n\nStep 3 - Create and train the model")
# create the model
inputs = keras.Input(shape=(8,))
hidden1 = Dense(12,activation='relu')(inputs)
hidden2 = Dense(8,activation='relu',)(hidden1)
output = Dense(1,activation='sigmoid')(hidden2)
model = keras.Model(inputs,output)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# summarize history for loss and accuracy as a function of the epochs
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color=color)
ax1.plot(history.history['loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color) # we already handled the x-label with ax1
ax2.plot(history.history['accuracy'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() # otherwise the right y-label is slightly clipped
plt.show()

X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))

print("\nExporting SavedModels: ")

# Keras 모델 저장
model.save('pima_model.keras')

# 모델 로드
model = keras.models.load_model('pima_model.keras')

# evaluate model
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))