import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('LeNet-5.h5')

# Prepare the calibration dataset (for calibrating the quantization range)
_, (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test[:100]  # only use first 100 datas
x_test = x_test.reshape(-1, 28, 28, 1)

# Create quantization converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Initial optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define calibration data generator
def representative_dataset():
    for data in x_test:
        yield [data.reshape(1, 28, 28, 1)]

# Set calibration dataset
converter.representative_dataset = representative_dataset

# Set target data type to int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Quantize input to int8
converter.inference_output_type = tf.int8  # Quantize output to int8

# convert
tflite_int8_model = converter.convert()

# Save
with open("LeNet-5-int8.tflite", "wb") as f:
    f.write(tflite_int8_model)

print("Model converted to int8 format.")
