
# Mengimpor TensorFlow dan modul yang diperlukan dari TensorFlow Transform (TFX)
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

# Mendefinisikan nama kunci untuk label (humor) dan fitur (text) dalam dataset.
LABEL_KEY = "humor"
FEATURE_KEY = "text"

# Fungsi untuk menghasilkan nama kunci fitur yang telah ditransformasi dengan menambahkan suffix "_xf".
def transformed_name(key):
    return key + "_xf"

# Fungsi preprocessing - menyiapkan transformasi yang diperlukan untuk data, seperti mengubah teks menjadi 
# huruf kecil dan memastikan label berada dalam format yang tepat
def preprocessing_fn(inputs):
    outputs = {}
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
