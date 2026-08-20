# Mind the Gap

`mindthegap` provides gap-filling functions for single-variable ocean color data
(e.g. chlorophyll) and multi-variable spectral data. It grew out of work started
by University of Washington Varanasi interns and developed further during GeoHackWeek 2024 and OceanHackWeek 2025
([proj_gap](https://github.com/oceanhackweek/ohw25_proj_gap)).

The primary model is a U-Net extended by SAFS Varanasi interns in 2026, as part of an UW eScience Acccelerator project in summer 2026.

## Pipeline

```mermaid
flowchart TB
    subgraph A[1. Data access]
        A1[Cloud data] --> A2[Subset region / time / variables]
        A2 --> A3[xarray dataset]
    end
    subgraph B[2. Batch generation]
        B1[Split: train / val / test]
        B2[Processing: standardize, masks, time vars]
        B3[Patch rules: size, overlap, variables]
        B4[xbatcher batch generator]
        B1 --> B4
        B2 --> B4
        B3 --> B4
    end
    subgraph C[3. Train model]
        C1[U-Net]
    end
    A3 --> B2
    B4 --> C1
```

## Quick links

- [Installation](installation.md)
- [API Reference](api.md)
- [Source on GitHub](https://github.com/SAFS-Varanasi-Internship/mindthegap)
