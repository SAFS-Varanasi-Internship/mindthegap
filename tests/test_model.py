from mindthegap.model import UNET_DEPTH, unet_spatial_multiple


def test_unet_spatial_multiple_matches_current_architecture():
    assert unet_spatial_multiple() == 2 ** UNET_DEPTH


def test_unet_spatial_multiple_derives_from_model_structure():
    import tensorflow as tf
    from tensorflow.keras import Input, layers

    def deep_unet(input_shape, verbose=False, **kwargs):
        inputs = Input(shape=input_shape)
        x = inputs
        for number_filters in (32, 64, 128, 256):
            x = layers.Conv2D(
                number_filters, 3, padding="same", activation="relu"
            )(x)
            x = layers.MaxPooling2D()(x)
        outputs = layers.Conv2D(1, 3, padding="same")(x)
        return tf.keras.Model(inputs, outputs)

    assert unet_spatial_multiple(build_fn=deep_unet) == 16


def test_unet_spatial_multiple_handles_strided_convolutions():
    import tensorflow as tf
    from tensorflow.keras import Input, layers

    def strided_unet(input_shape, verbose=False, **kwargs):
        inputs = Input(shape=input_shape)
        x = inputs
        for number_filters in (32, 64):
            x = layers.Conv2D(
                number_filters, 3, strides=2, padding="same", activation="relu"
            )(x)
        outputs = layers.Conv2D(1, 3, padding="same")(x)
        return tf.keras.Model(inputs, outputs)

    assert unet_spatial_multiple(build_fn=strided_unet) == 4


def test_unet_spatial_multiple_ignores_transpose_upsampling():
    import tensorflow as tf
    from tensorflow.keras import Input, layers

    def up_only(input_shape, verbose=False, **kwargs):
        inputs = Input(shape=input_shape)
        x = layers.Conv2DTranspose(8, 3, strides=2, padding="same")(inputs)
        outputs = layers.Conv2D(1, 3, padding="same")(x)
        return tf.keras.Model(inputs, outputs)

    assert unet_spatial_multiple(build_fn=up_only) == 2 ** UNET_DEPTH
