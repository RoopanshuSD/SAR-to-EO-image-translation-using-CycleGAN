# SAR-to-EO Image Translation using Lightweight CycleGAN + CBAM Attention

##  Team Members

- **Roopanshu Gupta**  
   roopanshugupta_se24b03_018@dtu.ac.in
- **Riddhima Bhargava**  
   riddhimabhargava_se24b03_011@dtu.ac.in

---

## 📌 Project Overview

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

## ▶ Instructions to Run Code

1. **Clone or Upload to Kaggle**
   - Upload all `.py` and `.ipynb` files along with this `README.md`.

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

## 🛠️ Data Preprocessing Steps

- Used custom `SARToEODataset` class:
  - Reads paired `.tif` images using `rasterio`
  - Extracts `VV`, `VH`, and VV/VH ratio channels from SAR
  - Selects RGB bands from EO images
  - Supports normalization: `dynamic`, `clip`, or `none`
- Images are resized to **256×256**
- SAR images are visualized in 3-channel format (VV, VH repeated)

---

## 🧱 Models Used

###  Generator
- Custom architecture using:
  - Depthwise Separable Convolutions
  - Inverted Residual Blocks
  - **CBAM attention block**
- Final activation: `Tanh` (to output normalized EO images)

###  Discriminator
- PatchGAN-style CNN with **Spectral Normalization**
- Optional support for depthwise blocks

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


