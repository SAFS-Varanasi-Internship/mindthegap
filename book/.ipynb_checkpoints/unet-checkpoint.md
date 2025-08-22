# What is U-Net?

U-Net is a Convolutional Neural Network (CNN) architecture. CNN is a type of deep learning model that is particularly effective for visual data and high dimensional data analysis. It is powerful in capturing spatial hierarchies and patterns, and is widely used in computer vision tasks.
U-Net follows an autoencoder architecture, where the encoder half down-samples input images progressively and extracts features, while the decoder half constructs predictions based on these features. It is effective and accurate with rather limited data.

## Model Architectures
The model uses three encoder layers of filter sizes 64, 128, 256, and three decoder layers of filter sizes 128, 64, 1. Each encoder block consists of two `Conv2D` layers, one `MaxPool2D` layer, and one `BatchNormalization` layer. Each decoder block consists of one `Conv2DTranspose` layer, one `Concatenate` layer, two `Conv2D` layer, and one `BatchNormalization` layer. The output of the final decoder layer is the gap-filled prediction of Chl-a.
- `Conv2D`: applies 2D convolution operations to the input. These layers are for feature detection (lines, edges, objects, patterns, etc.) in the encoder half, and for making predictions in the decoder half.
  `filters`: number of output channels and the number of features detected.
  `kernel_size`: size of the filters. All filters in this model are of size 3x3.
  `padding`: adds extra pixels to the input images. Padding of `same` ensures the same output dimensions as the input.
  `activation`: introduces non-linearity to neural networks that differentiate NNs from linear models. All layers other than the final layer uses 'ReLU', which outputs the input directly if positive and 0 if non-positive. The final layer uses 'Linear' due to potential negative values in log(Chl-a) predictions.
- `MaxPooling2D`: downsamples the input by taking the maximum in a given window (default is 2x2). It reduces complexity for future computations while retaining the most significant features. The output dimension is half of the input.
- `BatchNormalization`: normalizes the input. It reduces overfitting and improves the generalizability of a model.
- `Conv2DTranspose`: performs a "reverse" convolution and upsamples the input. The output dimension doubles the input.
- `Concatenate`: merges the upsampled feature maps with the feature maps from the corresponding encoder. It retains the higher-resolution features that were lost during downsampling.

## References
```{bibliography}
:filter: docname in docnames
```

