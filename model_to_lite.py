import tensorflow as tf

print("Starting the model conversion process...")

try:
    print("Loading the saved Keras model...")
    model = tf.keras.models.load_model('TFRS_LTSM_model')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

try:
    print("Configuring the TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Use Select TF Ops and disable lowering tensor list ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    print("TFLiteConverter configured successfully.")
except Exception as e:
    print(f"Error configuring the TFLiteConverter: {e}")
    exit(1)

try:
    print("Starting the conversion...")
    tflite_model = converter.convert()
    print("Model converted successfully.")
except Exception as e:
    print(f"Error during the conversion: {e}")
    exit(1)

try:
    print("Saving the converted model...")
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Converted model saved successfully.")
except Exception as e:
    print(f"Error saving the converted model: {e}")
    exit(1)

print("Model conversion process completed.")
