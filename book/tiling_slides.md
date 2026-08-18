---
marp: true
paginate: true
---

# The tiling artifact: a tile-to-cloud ratio

- Training on **small tiles** leaves a grid or seam texture in the fill. The **whole-frame model is clean**.
- Cause: **synthetic clouds cut at tile boundaries**. Ruled out: undertraining, the geo channels, the Conv2DTranspose checkerboard, and the loss location.
- It is a **ratio, not a fixed pixel size**: the artifact grows with cloud size and shrinks with tile size.
- Gone once the **tile is about 10x the cloud sigma** (about 128 px for the blob-12 clouds). The same **pixel** threshold held on IO.zarr (25 km) and Copernicus (4 km), so it **transfers across datasets**.
- Fix: tile at least about 10x blob_sigma, or **fit the whole frame** (no tiling, so no artifact).

![w:1000](https://github.com/user-attachments/assets/508b5327-cfb0-4da4-9eea-b150edb89494)

88 and 112 px show the grid; 128 px and whole-frame are clean (Oman, 4 km).

---

# When you have to tile, and how big: the limits

Two ceilings on the Hub (T4 GPU, batch 8, about 12 channels):

- **GPU** caps one image, a tile or a whole frame, at about **1000 px per side** (fit at 960, out of memory at 1200). The safe 128 px tile is about **8x under** it.
- **RAM** caps the in-memory cube at about **3.5 GB** = days x lat x lon x channels x 4 bytes. More days trades against a smaller box:

| days | max side (4 km) | region | cube |
|---|---|---|---|
| 30 | ~1560 px | ~65 deg | 3.5 GB |
| 45 | ~1270 px | ~53 deg | 3.5 GB |
| 90 | ~900 px | ~37 deg | 3.5 GB |
| 180 | ~640 px | ~27 deg | 3.5 GB |
| 365 | ~450 px | ~19 deg | 3.5 GB |

**Takeaway:** if the region fits a whole-frame pass there is no tiling and no artifact; if it is larger, tile at 10x the cloud or more.
