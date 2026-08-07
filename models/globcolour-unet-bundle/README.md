---
library_name: keras
tags:
  - gap-filling
  - ocean-color
---

# GlobColour U-Net gap filler

Keras model predicting **CHL** for **GlobColour**.

## Intended use

This model is intended for lat 25.687498092651367 to 30.979164123535156, lon 42.02083969116211 to 47.31250762939453
during 1997-09-04 to 1997-11-03.

## Inputs

- Channel 0: `masked_target`
- Channel 1: `masked_target_m1`
- Channel 2: `masked_target_p1`
- Channel 3: `day_sin`
- Channel 4: `day_cos`
- Channel 5: `synthetic_missing_flag`
- Channel 6: `true_missing_flag`
- Channel 7: `valid_masked_target_flag`
- Channel 8: `land_flag`

## Preprocessing

Expected input shape: `[None, None, None, 9]`

Transforms and standardization parameters are recorded in
`model_metadata.yaml`.

## Limitations

Validated only for the documented product, region, training period, channel order, and preprocessing configuration.

## Source

Repository: https://github.com/SAFS-Varanasi-Internship/mindthegap

Git commit: `df8d95c1ac84fc98b6e3368b671c04d2b578949e`
