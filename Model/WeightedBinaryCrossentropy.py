import tensorflow as tf
from keras.saving import saving_lib
import tensorflow.keras as keras
tf.keras.saving.get_custom_objects().clear()
@tf.keras.saving.register_keras_serializable(name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(target, output, weights):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_) 
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce
@tf.keras.saving.register_keras_serializable(name="WeightedBinaryCrossentropy")
class WeightedBinaryCrossentropy:
    def __init__(
        self,
        label_smoothing=0.0,
        weights = [1.0, 1.0],
        axis=-1,
        name="weighted_binary_crossentropy",
        fn = None,
    ):
        
        super().__init__()
        self.weights = weights # tf.convert_to_tensor(weights)
        self.label_smoothing = label_smoothing
        self.name = name
        self.fn = weighted_binary_crossentropy if fn is None else fn

    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        self.label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        def _smooth_labels():
            return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        y_true = tf.__internal__.smart_cond.smart_cond(self.label_smoothing, _smooth_labels, lambda: y_true)

        return tf.reduce_mean(self.fn(y_true, y_pred, self.weights),axis=-1)
    
    def get_config(self):
        config = {"name": self.name, "weights": self.weights, "fn": self.fn}

        # base_config = super().get_config()
        return dict(list(config.items()))

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).
        Args:
            config: Output of `get_config()`.
        """
        if saving_lib.saving_v3_enabled():
            fn_name = config.pop("fn", None)
            if fn_name:
                config["fn"] = get(fn_name)
        return cls(**config)