# SAR-to-EO Image Translation using Lightweight CycleGAN + CBAM Attention

##  Team Members

- **Roopanshu Gupta**  
   roopanshugupta_se24b03_018@dtu.ac.in
- **Riddhima Bhargava**  
   riddhimabhargava_se24b03_011@dtu.ac.in

---

##  Project Overview

This project focuses on **translating SAR (Synthetic Aperture Radar) images into Electro-Optical (EO)** images using a **custom-built lightweight CycleGAN architecture**. The goal is to generate realistic EO outputs from radar inputs, which is especially useful in remote sensing scenarios where EO data is missing due to clouds or night-time imaging.

Our model integrates **attention mechanisms (CBAM)** and **lightweight MobileNet-style building blocks** to strike a balance between **speed**, **performance**, and **generalizability**.

---

##  What Makes Our Project Unique

-  **Lightweight Generator Architecture**:
  - Built from scratch using **depthwise separable convolutions** and **inverted residual blocks**.
  - Significantly reduces computational overhead.

-  **CBAM Attention Integration**:
  - Channel and spatial attention enhances feature discrimination in both low- and high-level layers.

-  **Hybrid Multi-Loss Strategy**:
  - Combines **Charbonnier Loss**, **MS-SSIM**, **Perceptual Loss (VGG)**, and **GAN loss** to ensure texture, structure, and realism.

-  **End-to-End Modular Pipeline**:
  - Training, evaluation, checkpointing, and visualization are cleanly organized and automated.

---

##  Instructions to Run Code

1. **Clone or Upload to Kaggle**
   - Upload `.ipynb` files along with this `README.md`.

2. **Prepare Dataset**
   - Format: `/train/SAR`, `/train/EO`, `/val/SAR`, etc.
   - File type: `.tif` (multi-band images)
   - Ensure SAR and EO images are **paired and aligned**.

3. **Train the Model**
   - All training logic is provided in the notebook:
     ```
     Run all cells up to training
     Modify `num_epochs` as needed
     ```

4. **Evaluate**
   - Final metrics and visualizations will be shown at the end of training.

5. **Export Results**
   - Checkpoints are saved in `/checkpoints`
   - Download via zipped link at the end

---

##  Data Preprocessing Steps

We designed a robust and modular preprocessing pipeline tailored for paired **SAR-to-EO** image translation. The key component is our custom `SARToEODataset` class.

###  Dataset Class: `SARToEODataset`

This custom PyTorch `Dataset` handles all preprocessing operations:

#### 1. Paired Data Loading
- Loads SAR and EO images from respective folders.
- Assumes 1:1 pairing between SAR and EO `.tif` files.
- Recursively reads all `.tif` files using Python’s `glob` and `rasterio`.

#### 2.  SAR Image Processing
- SAR input is expected to contain **two bands**: VV and VH polarization.
- A **third ratio channel** is created: `VV / (VH + ε)` to add context on backscatter intensity differences.
- Final SAR tensor shape per sample: **[3, H, W]**

#### 3.  EO Image Processing
- EO images are read in their full multi-band format using `rasterio`.
- Based on `band_config`, appropriate bands are selected:
  - `'rgb'` → Red, Green, Blue
  - `'rgb_nir'` → Red, Green, Blue, NIR
  - `'nir_swir_red_edge'` → Custom configs supported
- Uses band descriptions (e.g., "red", "green") to map channel indices automatically.
- Fallback to default bands `[3, 2, 1]` (B4, B3, B2) if descriptions are unavailable.

#### 4.  Normalization Modes
Supports three flexible normalization strategies:
- `'dynamic'`: Normalize per image using its min/max, then scale to **[-1, 1]**
- `'clip'`: Clip using a specified range, scale to [0, 1], then to [-1, 1]
- `'none'`: No normalization (use raw values)

This ensures the model receives stable inputs even across varying image conditions.

#### 5.  Transformations
- All SAR and EO images are resized to **256×256** pixels for efficient training.
- During training:
  - **Random horizontal flipping** is applied for data augmentation.
- During validation/test:
  - Only resizing is applied (no augmentation).

#### 6.  Visualization Format
For qualitative evaluation:
- SAR channels `VV` and `VH` are repeated to **3 channels** (e.g., `[1, H, W] → [3, H, W]`) to simulate RGB for visualizations.
- Output images are denormalized to [0, 1] before display or saving.

---

This preprocessing ensures:
- Temporal and spatial alignment of SAR-EO pairs
- Robustness to varying intensity scales
- Compact and consistent inputs for the generator and discriminator


---

##  Models Used

Our model architecture is carefully designed for lightweight yet high-quality SAR-to-EO image translation. The two core components are the **Generator** and **Discriminator**, each incorporating modern design elements for performance and efficiency.

---

###  Generator

The generator is a **lightweight encoder-decoder** architecture enhanced with **attention mechanisms** for better spatial understanding and spectral reconstruction.

#### 🔧 Key Architectural Components:

1. **Initial Convolution**
   - Projects the 3-channel SAR input (VV, VH, VV/VH) to 32 feature maps.

2. **Downsampling Layers**
   - Two downsampling stages using:
     - **Depthwise Separable Convolutions**: Efficient alternative to standard convolutions.
     - Maintains spatial structure while reducing parameters and computation.

3. **Bottleneck Layers**
   - Multiple **Inverted Residual Blocks** inspired by MobileNetV2.
   - Each block expands, processes, and compresses features with residual connections.

4. **Attention Mechanism: CBAM**
   - Injected after bottleneck blocks.
   - **CBAM (Convolutional Block Attention Module)**:
     - Combines **Channel Attention** (focuses on “what”) and **Spatial Attention** (focuses on “where”).
     - Helps the generator prioritize important regions and channels for more accurate translation.

5. **Upsampling Layers**
   - Two stages of bilinear upsampling followed by Depthwise Separable Convolutions to restore spatial resolution.

6. **Output Layer**
   - A final convolution projects the features to 3 output EO bands.
   - Uses `Tanh` activation to produce outputs in the **[-1, 1]** range.

---

###  Discriminator

The discriminator follows the **PatchGAN** design, which judges the realism of **patches** instead of entire images—allowing it to enforce high-frequency correctness.

####  Architecture

1. **Layered CNN Blocks**
   - Each block:
     - 2D convolution with kernel size 4 and stride 2
     - Optionally uses **BatchNorm**
     - Followed by `LeakyReLU` activation

2. **Spectral Normalization**
   - Applied to each convolutional layer for **training stability** and to prevent discriminator overpowering the generator.

3. **Final Patch Output**
   - Outputs a **1-channel patch map** indicating real/fake confidence per patch.

---

##  Key Findings & Observations

- **CBAM enhances detail preservation** in EO predictions, especially in texture-sensitive regions.
- **Charbonnier loss** proved more stable than L1 in early epochs.
- **MS-SSIM + Perceptual Loss** improved structure and color matching.
- Lightweight model trained faster and generalized well even on small datasets.
- Best PSNR: **~19 dB** | Best SSIM: **~0.90**

---

##  Tools and Frameworks Used

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| **PyTorch**     | Core modeling and training       |
| **Torchvision** | Image processing utilities       |
| **matplotlib**  | Visualization                    |
| **pytorch_msssim** | MS-SSIM computation         |
| **rasterio**    | Reading multi-band satellite `.tif` images |
| **Kaggle**      | Notebook execution environment   |

---


