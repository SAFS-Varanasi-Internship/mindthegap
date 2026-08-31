"""TensorFlow input pipelines, model construction, and fitting."""

from dataclasses import replace

import numpy as np
import xarray as xr

# Fallback spatial multiple for :func:`UNet` inputs, used only if the factor
# cannot be derived by inspecting the built model. Each encoder stage halves
# the spatial dimensions, so inputs must be divisible by ``2 ** UNET_DEPTH``.
UNET_DEPTH = 3


def unet_spatial_multiple(build_fn=None):
    """Required spatial multiple for :func:`UNet` inputs.

    The U-Net halves each spatial dimension once per encoder downsampling
    stage, so every spatial dimension of the input must be divisible by the
    total downsampling factor for the decoder skip-connections to line up.

    The factor is derived by **inspecting the actual model** so it stays
    correct if :func:`UNet`'s architecture changes (e.g. more encoder stages
    are added). A small model is built and its downsampling layers'
    strides/pool sizes are multiplied together per spatial axis; the larger of
    the two axis factors is returned. If the model cannot be built (for example
    TensorFlow is unavailable) the function falls back to ``2 ** UNET_DEPTH``.

    ``build_fn`` builds the model to inspect; it defaults to :func:`UNet` and is
    mainly a testing seam.
    """
    if build_fn is None:
        build_fn = UNet
    try:
        import tensorflow as tf

        model = build_fn((None, None, 1), verbose=False)
    except Exception:
        return 2 ** UNET_DEPTH

    row_factor = 1
    col_factor = 1
    for layer in model.layers:
        # Only downsampling layers constrain the input size: pooling and
        # strided Conv2D. Conv2DTranspose upsamples, so it is excluded even
        # though it also carries a ``strides`` attribute.
        if isinstance(layer, tf.keras.layers.Conv2DTranspose):
            continue
        if isinstance(
            layer,
            (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D),
        ):
            strides = getattr(layer, "pool_size", None) or getattr(
                layer, "strides", None
            )
        elif isinstance(layer, tf.keras.layers.Conv2D):
            strides = getattr(layer, "strides", (1, 1))
        else:
            continue
        if not strides or tuple(strides) == (1, 1):
            continue
        row_factor *= int(strides[0])
        col_factor *= int(strides[1])

    factor = max(row_factor, col_factor)
    return factor if factor > 1 else 2 ** UNET_DEPTH


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
    from .validation import validate_options

    if verbose is None:
        verbose = options.verbose
    # make_generator needs the resolved split (which dates train/validate) and a
    # supported gridder; validate_options gives the how-to-fix guidance.
    validate_options(options, requires=["split", "gridder"])

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
        # resize-conv (bilinear upsample + conv) instead of Conv2DTranspose,
        # this avoids the checkerboard artifact the transposed convolution makes.
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        x = layers.Conv2D(
            number_filters,
            (3, 3),
            padding="same",
            activation="relu",
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

    # resize-conv (bilinear upsample + conv), same as the decoder loop above.
    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    x = layers.Conv2D(
        number_filters,
        (3, 3),
        padding="same",
        activation="relu",
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
    checkpoint_path=None,  # opt-in mid-training checkpoint (see docstring/below)
    verbose=None,
):
    """Fit a model using the training configuration on ``options``.

    ``options`` may be the full :class:`Options` object (``options.fit`` is
    used, and ``options.verbose`` sets the default verbosity) or a
    :class:`FitOptions` section directly. It supplies epochs, batch size,
    learning rate, patience, loss, and optimizer so these choices are not
    threaded individually through the pipeline. Pass ``checkpoint_path`` to also
    save the best model to disk during training, so a crash keeps progress.
    Returns the Keras ``History`` object.
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
        metrics=["mae"],
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
    # checkpoint the best model to disk during training so a crash keeps
    # progress. Opt-in via checkpoint_path; works alongside custom callbacks too.
    if checkpoint_path is not None:
        callbacks = list(callbacks) + [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
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


def gapfill_std(ds_std, model, options, *, time=None, verbose=None):
    """Run a trained model on an already-standardized dataset (low level).

    ``ds_std`` must be the output of
    :func:`mindthegap.prepare_model_data` in ``mode="gapfill"`` for the
    same model: it already carries the inference channels (real observations,
    ``estimate_flag`` over the cloud/NaN pixels to fill, ``unavailable_flag`` all
    zero) so **no relabelling is done here**. Channels are stacked in the exact
    order recorded in ``options.data.input_names`` and the model is run one time
    frame at a time.

    This is a **low-level** function: it returns the model prediction exactly as
    produced, in the model's standardized output space. It does **not** transform
    ``gapfilled_target`` in any way — it does not undo the target
    standardization (mean/std) and does not undo any log transform. Recovering
    the original units and removing the training standardization is the job of
    the higher-level ``gapfill(ds)`` (a later function), which will return
    ``gapfilled_<target>`` in the original units.

    Parameters
    ----------
    ds_std : xarray.Dataset
        Standardized gap-fill dataset (from ``prepare_model_data(...,
        mode="gapfill")``).
    model : keras.Model
        Loaded model, e.g. from :func:`mindthegap.load_model_bundle`.
    options : mindthegap.Options
        The resolved configuration (used only for the input channel order,
        ``options.data.input_names``).
    time : optional
        A single time label, a list/array of labels, or a slice selecting which
        frames to gap-fill. Defaults to every time step in ``ds_std``.
    verbose : bool, optional
        Print progress. Defaults to quiet.

    Returns
    -------
    xarray.Dataset
        A dataset with a single ``gapfilled_target`` variable of dimensions
        ``(time, lat, lon)`` (a ``time`` dimension of length one is retained),
        holding the raw model output in standardized space (no unstandardizing
        or de-logging applied).
    """
    from .validation import validate_options

    # gapfill_std needs the recorded channel order/standardization from a
    # training preparation pass; validate_options explains how to obtain them.
    validate_options(options, requires=["data_prepared"])
    names = list(options.data.input_names)
    missing = [name for name in names if name not in ds_std]
    if missing:
        raise KeyError(
            "ds_std is missing model input channels: " + ", ".join(missing)
        )

    selected = ds_std if time is None else ds_std.sel(time=time)
    if "time" not in selected.dims:
        selected = selected.expand_dims("time")

    frames = []
    times = selected["time"].values
    for index, stamp in enumerate(times):
        frame = selected.isel(time=index)
        values = np.stack(
            [np.nan_to_num(frame[name].values, nan=0.0) for name in names],
            axis=-1,
        ).astype("float32")
        prediction = np.asarray(
            model.predict(values[np.newaxis, ...], verbose=0)
        )
        if prediction.ndim != 4 or prediction.shape[0] != 1:
            raise ValueError(
                "model prediction must have shape (1, lat, lon, channels)"
            )
        # Return the raw model prediction as-is; do not unstandardize or de-log.
        frames.append(prediction[0, ..., 0])
        if verbose:
            print(f"gap-filled frame {index + 1}/{len(times)}: {stamp}")

    stacked = np.stack(frames, axis=0)
    gapfilled = xr.DataArray(
        stacked,
        dims=("time", "lat", "lon"),
        coords={
            "time": selected["time"],
            "lat": selected["lat"],
            "lon": selected["lon"],
        },
        name="gapfilled_target",
        attrs={
            "long_name": "gap-filled target (standardized model output)",
            "standardized": True,
        },
    )
    return gapfilled.to_dataset()
