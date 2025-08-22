# What is DINCAE?

DINCAE (Data-Interpolating Convolutional Auto-Encoder) {cite:p}`Barthetal2020, Barthetal2022` is a CNN encoder–decoder tailored to reconstruct gappy gridded data and provide pixel-wise uncertainty. Like U-Net, it uses a downsampling encoder, skip connections, and an upsampling decoder, but it differs in key ways: it is error-aware (ingests measurement uncertainty/quality via inverse variances and treats missing pixels explicitly) and uncertainty-estimating (trained with a heteroscedastic Gaussian likelihood to output both a mean field and a variance map). It also inserts fully connected layers at the bottleneck (not purely convolutional like classic U-Net) and typically uses average pooling and nearest-neighbor upsampling tuned for the data variable. U-Net is a general-purpose, fully convolutional image model that yields point estimates, whereas DINCAE is U-shaped but specialized for incorporating error information and returning calibrated uncertainties.

## How it’s different from the U-Net model we used

* Uncertainty-aware vs. point estimate. DINCAE is trained with a heteroscedastic Gaussian likelihood and predicts per-pixel variance; a vanilla U-Net typically uses L2 / cross-entropy and does not output uncertainties. 
* Error-aware inputs. DINCAE feeds the network the inverse error variance (precision) and treats missing data as infinite error; classic U-Net pipelines don’t encode measurement error this way. 
* Bottleneck design. DINCAE includes fully connected layers (with dropout) in the bottleneck; our U-Net is fully convolutional without dense layers. 
* Upsampling & pooling choices. DINCAE uses nearest-neighbor upsampling and finds average pooling better for SST; the original U-Net used up-convolutions (transposed conv) and max pooling. 
* Encoder/decoder mechanics: our U-Net uses MaxPooling and Conv2DTranspose upsampling. DINCAE tested max vs average pooling and proceeds with average pooling, and it upsamples via nearest-neighbor interpolation (not transposed convolutions). 
* Activations: our blocks use ReLU (linear at the end). DINCAE uses leaky ReLU in convolutions (α≈0.2) and standard ReLU in the dense layers. 


## References
```{bibliography}
:filter: docname in docnames
```

