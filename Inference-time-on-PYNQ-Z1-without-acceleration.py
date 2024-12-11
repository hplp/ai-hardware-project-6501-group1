import numpy as np
import tflite_runtime.interpreter as tflite
import pandas as pd
import time

# Load dataset
csv_file = "mnist_test.csv"
mnist_data = pd.read_csv(csv_file, header=None).values
y_test = mnist_data[:, 0].astype("int32")  # Lable in first column
x_test = mnist_data[:, 1:].astype("float32").reshape(-1, 28, 28, 1) / 255.0

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="LeNet-5-int8.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# prepare datasets
x_test = (x_test * 255 - 128).astype(np.int8)  # convert to INT8 [-128, 127]

# Start inference
start_time = time.time()
predicted_labels = []

for i in range(len(x_test)):
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1])

    # Perform inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_labels.append(np.argmax(output_data))

end_time = time.time()

# Performance
accuracy = np.mean(np.array(predicted_labels) == y_test)

print(f"Test accuracy: {accuracy:.2%}")
print(f"Total inference time: {end_time - start_time:.4f} seconds")
print(f"Average inference time per image: {(end_time - start_time) / len(x_test) * 1000:.4f} ms")
