import numpy as np
import time
import tensorflow as tf
import pandas as pd

# load file
csv_file = "/data/mnist_test.csv"
mnist_data = pd.read_csv(csv_file, header=None).values

# load test dataset
y_test = mnist_data[:, 0]   # label
x_test = mnist_data[:, 1:]  # pixels

# data preprocess
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_test = y_test.astype(np.int32)


model = tf.keras.models.load_model('../Models/LeNet-5.h5')

# inference time calculation
start_time = time.time()
predictions = model.predict(x_test)
end_time = time.time()

# Accuracy
predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == y_test)

print(f"Total inference time for the test set: {end_time - start_time:.4f} seconds")
print(f"Average inference time per image: {(end_time - start_time) / len(x_test) * 1000:.4f} ms")
print(f"Test set accuracy: {accuracy:.2%}")
