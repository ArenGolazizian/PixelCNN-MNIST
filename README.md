# PixelCNN on MNIST: Autoregressive Image Generation

## Project Overview
This project implements an autoregressive generative model (PixelCNN) on the MNIST dataset of handwritten digits. The goal is to model the distribution of pixel intensities and generate new, realistic digit images pixel by pixel.

**NOTE:** All the implementation and experiments for the PixelCNN model on the MNIST dataset are located at `/notebooks/PixelCNN-MNIST.ipynb`.

## Mathematical Background
Autoregressive models factorize the joint probability distribution into a product of conditional distributions:

$$ P(x) = \prod_i P(x_i | x_{1:i-1}) $$

Where each pixel is conditioned on all previous pixels in raster scan order (top to bottom, left to right).
The structure can be visualized as:

<div align="center">
    <img src="https://camo.githubusercontent.com/2f581257c289298057989d11aa1ad507c2af397b2471c592f7b17a5dbecd731e/687474703a2f2f736572676569747572756b696e2e636f6d2f6173736574732f323031372d30322d32322d3138333031305f343739783439345f7363726f742e706e67" alt="PixelCNN Architecture" width="60%">
</div>

## Model Architecture
The PixelCNN architecture uses masked convolutions to enforce the autoregressive property:

- **Masked Convolutions**: Two types of masks ensure proper conditioning:
    - **Mask A**: Blocks current and future pixels (used in first layer)
    - **Mask B**: Blocks only future pixels (used in subsequent layers)

- **Network Structure**:
    - Input → Mask A 7×7 conv → 6×(Mask B 7×7 conv + ReLU) → 1×1 conv → Sigmoid output
    - Hidden channels: 64

![PixelCNN Architecture](https://lilianweng.github.io/lil-log/assets/images/pixel-cnn.png)

The key innovation is that each pixel only has access to previously generated pixels, preserving the autoregressive property during both training and generation.

## Training Process
- **Data Prep**: MNIST, binarized via threshold 0.5.  
- **Optimizer**: Adam (lr=1e-3)  
- **Loss**: Binary Cross-Entropy  
- **Batch size**: 64, **Epochs**: 10  
- **Evaluation**: Track training and test loss each epoch.

## Results
The model was trained for 10 epochs, with careful monitoring of both training and test loss.
| Epoch | Train Loss | Test Loss |
|------:|-----------:|----------:|
|    10 | 0.0761     | 0.0764    |

Training loss decreased from 0.1243 to 0.0761, while test loss improved from 0.0864 to 0.0764, indicating good generalization without overfitting.

<div align="center">
    <img src="results/loss_curve.png" alt="Training and Testing Loss" width="80%">
</div>


**Generated samples:**

<div align="center">
    <img src="results/samples.png" alt="Generated Samples" width="80%">
</div>

The model quickly converges in early epochs and produces recognizable digit shapes, indicating strong autoregressive modeling of pixel dependencies.

## References
- [van den Oord et al., "Conditional Image Generation with PixelCNN Decoders," 2016, arXiv:1606.05328](https://arxiv.org/abs/1606.05328)  
- [LeCun et al., "MNIST Handwritten Digit Database," 1998](http://yann.lecun.com/exdb/mnist/)  
