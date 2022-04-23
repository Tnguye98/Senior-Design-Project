# Import TensorFlow
import tensorflow as tf
print(tf.__version__)

x = tf.range(-100, 100, 4)
x
y = x + 10
y

# Set random seed
tf.random.set_seed(42)

# 1. Create a model using the Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# 2. Compile the model
model.compile(loss = tf.keras.losses.mae,             # mae is short for mean absolute error
              optimizer = tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
              metrics = ["mae"])

# 3. Fit the model
model.fit(tf.expand_dims(x, axis =-1), y, epochs = 100, verbose = 1)