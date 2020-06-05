import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense

model = tf.keras.Sequential(
    [Input(shape=(1,)), Dense(units=10, activation="relu"), Dense(units=1)]
)
model.compile(loss=tf.keras.losses.mse)

x_train = np.arange(-10, 10, 0.001)
y_train = x_train ** 2

x_test = np.arange(-1, 1, 0.1)
y_test = x_test ** 2

# First, let's create a training Dataset instance.
# For the sake of our example, we'll use the same MNIST data as before.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Now we get a test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

# Since the dataset already takes care of batching,
# we don't pass a `batch_size` argument.
model.fit(train_dataset, epochs=3)
