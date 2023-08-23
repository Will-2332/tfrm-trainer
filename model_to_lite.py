import tensorflow as tf

model = tf.keras.models.load_model('TFRS_LTSM_model')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Use Select TF Ops and disable lowering tensor list ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
