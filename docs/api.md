# API Reference

## Data preparation

::: mindthegap.data
    options:
      members:
        - demo_data
        - prepare_model_data
        - crop_to_multiple
        - synthetic_cloud_cube
        - train_validation_dates
        - set_up_train_split_options

## Evaluation

::: mindthegap.evaluation

## Cloud banks

::: mindthegap.cloud_bank

## Model

::: mindthegap.model
    options:
      members:
        - UNet
        - make_xbatcher
        - make_generator
        - make_tf_gen
        - fit_model
        - gapfill_std

## Training

::: mindthegap.training
    options:
      members:
        - train_model
        - TrainingResult

## Model bundles

::: mindthegap.model_bundle
    options:
      members:
        - save_model_bundle
        - load_model_bundle
        - load_bundle_metrics

## Gridder

::: mindthegap.gridder
    options:
      members:
        - set_up_gridder_options
        - GridderRecommendation

## Session

::: mindthegap.session
    options:
      members:
        - set_up

## Options

::: mindthegap.options
    options:
      members:
        - Options
        - DataOptions
        - GridderOptions
        - FitOptions
        - SplitOptions
        - cloud_options_for
