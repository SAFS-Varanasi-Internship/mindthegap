"""TensorFlow input pipelines, model construction, and fitting."""

from dataclasses import replace

import numpy as np


def make_tf_gen(batcher, x_vars, label="full_target"):
    """Create a TensorFlow generator from standardized xbatcher blocks."""

    def gen():
        for batch in batcher:
            batch = batch.load()
            for time_index in range(batch.sizes["time"]):
                x = np.stack(
                    [
                        np.nan_to_num(
                            batch[var].isel(time=time_index).values,
                            nan=0.0,
                        )
                        for var in x_vars
                    ],
                    axis=-1,
                ).astype(np.float32)
                y = np.nan_to_num(
                    batch[label].isel(time=time_index).values,
                    nan=0.0,
                ).astype(np.float32)[..., np.newaxis]
                yield x, y

    return gen


def make_xbatcher(
    ds,
    patch_dims=None,
    overlap=None,
    preload_batch=False,
    options=None,
):
    """Create an xbatcher generator over time and spatial windows.

    Either pass ``patch_dims``/``overlap`` explicitly or pass a
    :class:`GridderOptions` as ``options`` to derive them.
    """
    import xbatcher as xb

    if options is not None:
        patch_dims = options.patch_dims()
        overlap = options.overlap_dims()
        preload_batch = options.preload_batch
    if patch_dims is None:
        raise ValueError("provide either patch_dims or options")

    kwargs = {
        "ds": ds,
        "input_dims": dict(patch_dims),
        "preload_batch": preload_batch,
    }
    if overlap is not None:
        kwargs["input_overlap"] = dict(overlap)
    return xb.BatchGenerator(**kwargs)


def make_generator(ds_std, options, *, verbose=None):
    """Build train/validation TensorFlow datasets from a standardized dataset.

    ``options`` is the full :class:`Options` object. The training and validation
    dates come from ``options.split``, the tiling from ``options.gridder``, the
    channel order/target from ``options.data``, and batching from
    ``options.fit``. Returns ``(train_dataset, val_dataset, train_steps,
    val_steps)`` so the notebook does not manage the intermediate ``ds_train`` /
    ``ds_val`` or step counts. ``verbose`` defaults to ``options.verbose``.
    """
    import tensorflow as tf

    if verbose is None:
        verbose = options.verbose
    if not options.split.is_resolved():
        raise ValueError(
            "options.split has no dates; call mtg.train_validation_dates first"
        )
    if options.gridder.method != "xbatcher":
        raise ValueError(
            f"Unsupported gridder method {options.gridder.method!r}; "
            "only 'xbatcher' is currently supported"
        )

    ds_train = ds_std.sel(time=options.split.train_selection())
    ds_val = ds_std.sel(time=options.split.val_selection())

    # Selected dates may be sparse (random) or a short manual window, so cap the
    # temporal patch length at the number of available time steps in each split.
    train_gridder = replace(
        options.gridder,
        time_chunk=min(options.gridder.time_chunk, ds_train.sizes["time"]),
    )
    val_gridder = replace(
        options.gridder,
        time_chunk=min(options.gridder.time_chunk, ds_val.sizes["time"]),
    )
    train_batcher = make_xbatcher(ds_train, options=train_gridder)
    val_batcher = make_xbatcher(ds_val, options=val_gridder)

    num_channels = len(options.data.input_names)
    tile_lat, tile_lon = options.gridder.tile_size
    output_signature = (
        tf.TensorSpec(
            shape=(tile_lat, tile_lon, num_channels), dtype=tf.float32
        ),
        tf.TensorSpec(shape=(tile_lat, tile_lon, 1), dtype=tf.float32),
    )

    train_dataset = (
        tf.data.Dataset.from_generator(
            make_tf_gen(
                train_batcher,
                options.data.input_names,
                label=options.data.target,
            ),
            output_signature=output_signature,
        )
        .shuffle(
            options.fit.shuffle_buffer,
            seed=options.resolved_shuffle_seed(),
            reshuffle_each_iteration=True,
        )
        .batch(options.fit.batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_generator(
            make_tf_gen(
                val_batcher,
                options.data.input_names,
                label=options.data.target,
            ),
            output_signature=output_signature,
        )
        .batch(options.fit.batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    train_steps = max(
        1,
        (len(train_batcher) * train_gridder.time_chunk)
        // options.fit.batch_size,
    )
    val_steps = max(
        1,
        (len(val_batcher) * val_gridder.time_chunk)
        // options.fit.batch_size,
    )
    if verbose:
        print(
            "Streaming datasets ready "
            f"(batch size {options.fit.batch_size}): "
            f"steps per epoch train={train_steps}, val={val_steps}"
        )
    return train_dataset, val_dataset, train_steps, val_steps


def UNet(input_shape, verbose=None, tile_size=None, input_names=None):
    """Build the fully convolutional U-Net used by the fitting notebook.

    ``input_shape`` is the Keras input shape ``(height, width, channels)``;
    height/width may be ``None`` for the fully-convolutional model.

    When ``verbose`` is true, a short summary of the model's expected input and
    output shapes is printed. ``verbose`` defaults to ``True``; pass an
    :class:`Options` object's ``options.verbose`` to tie this to the global
    verbosity. ``tile_size`` (a ``(lat, lon)`` pair) is used only to make the
    printed shapes concrete; when omitted the spatial dims from ``input_shape``
    are shown. ``input_names`` are printed as the channel order when provided.
    """
    import tensorflow as tf
    from tensorflow.keras import Input, layers

    inputs = Input(shape=input_shape)
    x = inputs
    filters = [64, 128, 256]
    encoder_images = []

    for number_filters in filters:
        encoder_images.append(x)
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNormalization()(x)

    for number_filters, encoder_image in zip(
        filters[:-1][::-1],
        encoder_images[::-1][:-1],
    ):
        x = layers.Conv2DTranspose(
            number_filters,
            3,
            2,
            padding="same",
        )(x)
        x = layers.concatenate([x, encoder_image])
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(
        number_filters,
        3,
        2,
        padding="same",
    )(x)
    x = layers.concatenate([x, encoder_images[0]])
    x = layers.Conv2D(
        number_filters,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)
    outputs = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="linear",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="U-net")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    if verbose is None:
        verbose = True
    if verbose:
        num_channels = input_shape[-1]
        if tile_size is not None:
            tile_lat, tile_lon = tile_size
        else:
            tile_lat, tile_lon = input_shape[0], input_shape[1]
        print(
            f"\nModel input shape: (batch, {tile_lat}, {tile_lon}, "
            f"{num_channels})"
        )
        print(f"Model output shape: (batch, {tile_lat}, {tile_lon}, 1)")
        if input_names is not None:
            print(f"\nInput names: {list(input_names)}")

    return model


def fit_model(
    model,
    train_data,
    options,
    *,
    validation_data=None,
    steps_per_epoch=None,
    validation_steps=None,
    callbacks=None,
    verbose=None,
):
    """Fit a model using the training configuration on ``options``.

    ``options`` may be the full :class:`Options` object (``options.fit`` is
    used, and ``options.verbose`` sets the default verbosity) or a
    :class:`FitOptions` section directly. It supplies epochs, batch size,
    learning rate, patience, loss, and optimizer so these choices are not
    threaded individually through the pipeline. Returns the Keras ``History``
    object.
    """
    import tensorflow as tf

    from .options import Options as _Options

    options_verbose = None
    if isinstance(options, _Options):
        options_verbose = options.verbose
        options = options.fit
    if verbose is None:
        verbose = 1 if (options_verbose is None or options_verbose) else 0

    optimizers = {
        "adam": tf.keras.optimizers.Adam,
    }
    if options.optimizer not in optimizers:
        raise ValueError(
            f"Unsupported optimizer {options.optimizer!r}; "
            f"choose from: {', '.join(optimizers)}"
        )
    model.compile(
        optimizer=optimizers[options.optimizer](
            learning_rate=options.learning_rate
        ),
        loss=options.loss,
        jit_compile=False,
    )

    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=options.patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        ]

    if verbose:
        print(
            "Starting training: "
            f"epochs={options.epochs}, batch_size={options.batch_size}, "
            f"patience={options.patience}"
        )

    history = model.fit(
        train_data,
        epochs=options.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )

    if verbose and "val_loss" in history.history:
        print(
            f"Best val_loss: {min(history.history['val_loss']):.6f}; "
            f"final train_loss: {history.history['loss'][-1]:.6f}"
        )
    return history
