"""Masked losses/metrics for self-supervised gap-fill training.

EDIT (item 1, masked training loss): the training target is 3 channels,
``[value, train_mask, estimate_mask]`` (built in ``make_tf_gen``):

- ``value``         : ``full_target``, the true standardized field (NaN -> 0).
- ``train_mask``    : ``observed_flag OR estimate_flag`` -- pixels that had a real
                      observation at time t (still-visible observed pixels plus the
                      synthetically-clouded ones). These are the pixels the
                      training loss scores. Real clouds and land (value 0, unknown)
                      are excluded, so the model is not trained to emit 0 there.
- ``estimate_mask`` : ``estimate_flag`` only -- the synthetic clouds. The
                      evaluation metrics score only these, i.e. how well the model
                      fills gaps it did not see (held-out gap-fill skill).

Registered as Keras serializables so a compiled model saves cleanly; inference
loads with ``compile=False`` (see ``load_model_bundle``) so these are not needed
to load a model for gap-filling.
"""
import tensorflow as tf
import keras

_EPS = 1e-6


@keras.saving.register_keras_serializable(package="mindthegap")
def masked_mse(y_true, y_pred):
    """Training loss: MSE over valid-CHLA pixels (observed or synthetic-cloud)."""
    value = y_true[..., 0:1]
    mask = y_true[..., 1:2]
    squared = tf.square(y_pred - value) * mask
    return tf.reduce_sum(squared) / (tf.reduce_sum(mask) + _EPS)


@keras.saving.register_keras_serializable(package="mindthegap")
def fakecloud_mse(y_true, y_pred):
    """Eval metric: MSE under the synthetic clouds only (held-out gap-fill)."""
    value = y_true[..., 0:1]
    mask = y_true[..., 2:3]
    squared = tf.square(y_pred - value) * mask
    return tf.reduce_sum(squared) / (tf.reduce_sum(mask) + _EPS)


@keras.saving.register_keras_serializable(package="mindthegap")
def fakecloud_mae(y_true, y_pred):
    """Eval metric: MAE under the synthetic clouds only (held-out gap-fill)."""
    value = y_true[..., 0:1]
    mask = y_true[..., 2:3]
    absolute = tf.abs(y_pred - value) * mask
    return tf.reduce_sum(absolute) / (tf.reduce_sum(mask) + _EPS)
