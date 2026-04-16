# 📷 Sheet 01 — Advanced Image Processing & Optimization

This project explores more advanced concepts in computer vision, including **image denoising, optimization, integral images, and filter separability**. It combines both theoretical understanding and practical implementation using OpenCV and NumPy.

The focus is on implementing core algorithms from scratch, evaluating their performance, and understanding their computational and mathematical foundations.

---

## 🚀 Features

* 🧠 Theoretical analysis of convolution properties
* 🧹 Image denoising using multiple filtering techniques
* ⚙️ Custom implementations of filters (from scratch)
* 📊 Image quality evaluation using PSNR
* 🔍 Parameter optimization for best denoising performance
* ⚡ Integral image computation for efficient region queries
* 🧮 Complexity analysis and runtime comparison
* 🧩 Filter separability analysis using SVD

---

## 🛠️ Tech Stack

* Python 3.12
* OpenCV 4.11
* NumPy 2.3.3
* matplotlib
* scikit-image (only for PSNR computation)

> ⚠️ Designed for Linux-based environments.
> ⚠️ Only the specified libraries are used.

---

## 📂 Project Structure

```
.
├── bonn.jpg
├── bonn_noisy.jpg
├── q2_denoising.py
├── q3_integral.py
├── q4_separability.py
├── report.pdf
├── README.md
```

---

## ⚙️ Installation

```
pip install opencv-python numpy matplotlib scikit-image
```

---

## ▶️ Usage

Run each task independently:

```
python q2_denoising.py
python q3_integral.py
python q4_separability.py
```

---

## 🧪 Implemented Tasks

### 1. Convolution Theorem (Theory)

* Proof of distributive property (discrete-time)
* Proof of associative property (continuous-time)

---

### 2. Image Denoising & Optimization

* Convert images to grayscale
* Apply and compare:

  * Gaussian Filter
  * Median Filter
  * Bilateral Filter

Each filter is implemented in two ways:

* Using OpenCV built-in functions
* Custom implementation from scratch

#### 📊 Image Quality Evaluation

* Compute PSNR between original and denoised images
* Compare performance across filters

#### ⚙️ Parameter Optimization

* Iterate over filter parameters:

  * Kernel size (Gaussian, Median)
  * σ values (Bilateral)
* Select parameters that maximize PSNR
* Use optimal values for final results

#### 💬 Discussion

* Analyze effectiveness of filters on mixed noise
* Compare theoretical expectations vs experimental results

---

### 3. Integral Images

* Compute integral image
* Efficiently calculate mean intensity of a region using integral image
* Compare with naive approach

#### ⚡ Performance Analysis

* Theoretical complexity comparison (Big-O)
* Empirical runtime measurement

---

### 4. Filter Separability

* Decompose kernels using SVD
* Determine separability of filters

#### 🔍 Approximation

* Approximate non-separable filters using largest singular value
* Apply both original and approximated filters

#### 📉 Error Analysis

* Compute pixel-wise absolute difference
* Report maximum error

---

## 📌 Notes

* All implementations follow strict library constraints
* Emphasis on both theoretical understanding and practical efficiency
* Code is structured to clearly separate each task
* This project builds directly on foundational OpenCV knowledge
