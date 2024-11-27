import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# LeNet-5 model
model = models.Sequential([
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'),  # C1
    layers.AveragePooling2D((2, 2)),  # S2
    layers.Conv2D(16, (5, 5), activation='relu', padding='valid'),  # C3
    layers.AveragePooling2D((2, 2)),  # S4
    layers.Flatten(),  # Flatten
    layers.Dense(120, activation='relu'),  # C5
    layers.Dense(84, activation='relu'),  # F6
    layers.Dense(10, activation='softmax')  # Output layer
])

# Model compiling
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# The structure of model
model.summary()

# train model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Model evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Transform to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite
with open('leNet-5.tflite', 'wb') as f:
    f.write(tflite_model)