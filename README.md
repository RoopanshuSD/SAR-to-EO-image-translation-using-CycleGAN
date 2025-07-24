# SAR-to-EO Image Translation using Lightweight CycleGAN + CBAM Attention

## Team Members

- **Roopanshu Gupta**  
  roopanshugupta_se24b03_018@dtu.ac.in  
- **Riddhima Bhargava**  
  riddhimabhargava_se24b03_011@dtu.ac.in  

---

## Project Overview

This project focuses on **translating SAR (Synthetic Aperture Radar) images into Electro-Optical (EO)** images using a **custom-built lightweight CycleGAN architecture**. The objective is to generate realistic EO outputs from radar inputs, especially in remote sensing scenarios where EO data is unavailable due to clouds or low-light conditions.

Our approach is validated across **three EO translation settings**:
1. **SAR → EO (RGB)**: B4, B3, B2  
2. **SAR → EO (NIR, SWIR, Red Edge)**: B8, B11, B5  
3. **SAR → EO (RGB + NIR)**: B4, B3, B2, B8 with NDVI integration

We leverage a **CBAM-attentive CycleGAN model** built with **MobileNet-style lightweight blocks** to balance **efficiency, performance, and spectral sensitivity**.

---

## What Makes Our Project Unique

- **Multiple Spectral Target Variants**  
  Supports EO image synthesis in **RGB**, **NIR-SWIR-RedEdge**, and **RGB+NIR (with NDVI)** formats.

- **Lightweight Generator Architecture**  
  Built from scratch using **depthwise separable convolutions** and **inverted residual blocks**, ensuring faster training and fewer parameters.

- **CBAM Attention Integration**  
  Integrates both **channel** and **spatial attention** to prioritize key features and regions during translation.

- **Hybrid Multi-Loss Strategy**  
  Combines:
  - Charbonnier loss for robustness  
  - MS-SSIM for structure preservation  
  - VGG-based perceptual loss for texture consistency  
  - GAN loss for realism

- **NDVI Integration in RGB+NIR Variant**  
  NDVI maps are derived from predicted Red and NIR bands to validate ecological consistency of generated EO outputs.

---

## Instructions to Run Code

1. **Upload Notebook to Kaggle**
   - Include the `.ipynb` notebook and `README.md`.

2. **Prepare Dataset**
   - Folder structure:
     ```
     /train/SAR
     /train/EO
     /val/SAR
     /val/EO
     /test/SAR
     /test/EO
     ```
   - File format: `.tif`  
   - Ensure files are paired by name.

3. **Choose Output Configuration**
   - Set `band_config` to one of:
     - `'rgb'`
     - `'rgb_nir'`
     - `'nir_swir_red_edge'`

4. **Train the Model**
   - Run all notebook cells
   - Modify `num_epochs` to control training length

5. **Evaluate**
   - Final PSNR, SSIM, and qualitative visualizations are shown after training

6. **Export Results**
   - Trained checkpoints are saved to `/checkpoints` and zipped for download

---

## Data Preprocessing Steps

We implemented a custom `SARToEODataset` class that dynamically adapts to the spectral configuration selected for EO.

### 1. Paired Data Loading
- Loads aligned `.tif` files from SAR and EO directories.
- Uses `rasterio` for robust multi-band reading.
- Ensures 1:1 mapping across SAR and EO samples.

### 2. SAR Image Processing
- Inputs: **VV** and **VH** polarizations  
- Computes additional ratio channel: **VV / (VH + ε)**
- Final input shape: **[3, H, W]**

### 3. EO Band Selection
Supports three `band_config` settings:

- `'rgb'` → Red, Green, Blue (B4, B3, B2)
- `'nir_swir_red_edge'` → NIR, SWIR, Red Edge (B8, B11, B5)
- `'rgb_nir'` → Red, Green, Blue, NIR (B4, B3, B2, B8)

The dataset dynamically maps band descriptions like "red", "nir" using `rasterio` metadata. If unavailable, it falls back to hardcoded band indices.

### 4. NDVI Computation (only for RGB+NIR variant)
- For `band_config='rgb_nir'`, NDVI is computed during evaluation as:  
  `NDVI = (NIR - Red) / (NIR + Red + ε)`
- Used to assess vegetation integrity in EO reconstructions.

### 5. Normalization
Three supported modes:
- `'dynamic'`: Min-max per image, scaled to [-1, 1]
- `'clip'`: Range-based clipping, scaled to [-1, 1]
- `'none'`: No normalization

### 6. Transformations
- All inputs resized to **256×256**
- Data augmentation during training: **random horizontal flip**
- Only resizing applied during validation/test

### 7. Visualization
- SAR: Channels repeated to simulate RGB-style 3-channel input
- EO outputs denormalized to [0, 1] before visualization

---

## Models Used

Both generator and discriminator are modular and lightweight to ensure generalizability across spectral settings.

### Generator

A lightweight encoder-decoder architecture with integrated CBAM attention.

#### Key Components:
1. **Input Projection**
   - Converts 3-band SAR input into a 32-channel representation

2. **Downsampling Layers**
   - Uses **depthwise separable convolutions** with stride 2
   - Retains spatial information with minimal parameter cost

3. **Bottleneck**
   - Stack of **inverted residual blocks** inspired by MobileNetV2
   - Efficient expansion + projection architecture

4. **CBAM Attention Module**
   - Injected after bottleneck
   - Channel attention using global pooling
   - Spatial attention using convolutional feature maps

5. **Upsampling**
   - Bilinear upsampling followed by depthwise separable convolutions

6. **Output Layer**
   - Final convolution projects to:
     - 3 channels for RGB or NIR-SWIR-RE
     - 4 channels for RGB+NIR
   - `Tanh` activation outputs normalized EO image in [-1, 1]

### Discriminator

PatchGAN-style CNN for fine-grained spatial discrimination.

#### Features:
- 4×4 convolution blocks with LeakyReLU and optional BatchNorm
- **Spectral normalization** for GAN stability
- Outputs a patch-wise prediction map

---

## Key Findings and Observations

- The model adapts well to multiple EO output types with minimal changes.
- **RGB+NIR variant produces realistic NDVI patterns**, indicating biologically plausible EO synthesis.
- **CBAM improves detail** in spatially complex regions (e.g., urban textures or vegetation edges).
- **Charbonnier loss** outperforms L1 loss in early stability and final sharpness.
- **MS-SSIM and perceptual losses** significantly help with structure and color alignment.
- Our lightweight model achieves strong generalization on small training datasets.

| Configuration         | PSNR (avg) | SSIM (avg) |
|-----------------------|------------|-------------|
| SAR → EO (RGB)        | ~19 dB     | ~0.6        |
| SAR → EO (RGB+NIR)    | ~16.7 dB   | ~0.45        |
| SAR → EO (NIR-SWIR-RE)| ~18 dB   | ~0.6        |

---

## Tools and Frameworks Used

| Tool / Library       | Purpose                                 |
|----------------------|------------------------------------------|
| **PyTorch**          | Core deep learning framework             |
| **Torchvision**      | Transforms, image saving utilities       |
| **matplotlib**       | Training/validation curve plotting       |
| **pytorch_msssim**   | MS-SSIM metric calculation               |
| **rasterio**         | Multiband satellite image reading        |
| **Kaggle Notebooks** | Execution environment with GPU support   |



