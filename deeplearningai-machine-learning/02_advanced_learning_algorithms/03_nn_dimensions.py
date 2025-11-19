import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
# numpy typing
from numpy.typing import NDArray
from typing import Annotated

import pandas as pd

import logging

import tensorflow as tf
from tensorflow.keras.layers import Normalization, Dense, Input
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy # type: ignore

from lab_coffee_utils import load_coffee_data

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

print(f'Tensorflow version : {tf.__version__}')
print(f'Using GPU: {"no" if len(tf.config.list_physical_devices("GPU")) == 0 else "yes"}.')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

num_samples = 200
num_features = 2


features = ['temperature', 'duration']

X: Annotated[NDArray[np.float64], (num_samples, num_features)]
# Y is a 1D matrix rather than a vector!
Y: Annotated[NDArray[np.float64], (num_samples, 1)]
X, Y = load_coffee_data()

assert X.shape[0] == Y.shape[0], f'X and Y should have the same number of samples (.shape[0]), but got {X.shape[0]} for X and {Y.shape[0]} for Y'
assert X.shape[1] == num_features, f'X should have {len(features)} features, but got {X.shape[1]}'
assert Y.shape[1] == 1, f'Y should have 1 output column, but got {Y.shape[1]}'

X_df = pd.DataFrame(X, columns=features)
print()
print(f'X shape: {X.shape}')
print()
print(X_df.head())
print()
Y_df = pd.DataFrame(Y, columns=['well_roasted'])
print()
print(f'Y shape: {Y.shape}')
print()
print(Y_df.head())
print()

# Normalization (no shape change)
normalizer = Normalization(axis=-1)
normalizer.adapt(X)

Xn = normalizer(X);


# Fake more examples
Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000, 1))

print(f'Xt shape: {Xt.shape}\nYt shape: {Yt.shape}')

# Create the neural network
# Using a linear output and from_logits=True in the loss to improve numerical stability
model: Model = Sequential([
    tf.keras.Input(shape=(2,), name='input'),
    Dense(3, activation='sigmoid', name='l1'),
    Dense(1, activation='linear', name = 'l2')
])

model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

print(model.summary())

print(f'The model has {len(model.layers)} layers.\n')

print('Layer 1')
l1 = model.get_layer('l1')
W1_l1, b1_l1 = l1.get_weights()
n_units = l1.get_config()['units']
print(f'W shape = {W1_l1.shape} (number_of_features, number_of_neurons)')
print(f'b shape  = {b1_l1.shape} (number_of_neurons)')
print(f'W1_l1[0] == {n_units} values going in the {n_units} neurons of the layer `{l1.name}` FOR THE FIRST FEATURE -> {W1_l1[0]}')
print(f'W1_l1[1] == {n_units} values going in the {n_units} neurons of the layer `{l1.name}` FOR THE SECOND FEATURE -> {W1_l1[1]}\n')

print(f'b: there\'s only 1 bias per value in `y^ = np.dot(W, X) + b`; b is always a scalar, so there\'s only 1 b per neuron')
