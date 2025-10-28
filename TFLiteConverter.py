import tensorflow as tf

# Carica il modello Keras dal file .h5
model = tf.keras.models.load_model("model.h5")

# Passa il modello al converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Salva il modello TFLite
with open("model.tflite", "wb") as f:
    f.write(tflite_model)