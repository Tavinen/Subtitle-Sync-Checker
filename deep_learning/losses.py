# losses.py
from hybrid_config import HYBRID_PARAMS
import tensorflow as tf

def hybrid_loss(y_true, y_pred):
    return (HYBRID_PARAMS["audio_weight"] *
           tf.keras.losses.binary_crossentropy(y_true, y_pred) +
           (1 - HYBRID_PARAMS["audio_weight"]) *
           tf.keras.losses.binary_crossentropy(y_true, y_pred))