# 🚀 Denoising Diffusion Probabilistic Model (DDPM) from Scratch

This project implements a **Denoising Diffusion Probabilistic Model
(DDPM)** from scratch using **PyTorch**. It focuses on understanding the
core concepts behind diffusion models and building the complete pipeline
without relying on pre-built frameworks.

------------------------------------------------------------------------

## 📌 Overview

Diffusion models are a powerful class of generative models that learn to
generate data by reversing a gradual noising process.

In this project, we: - Start with pure noise - Train a neural network to
**denoise step-by-step** - Finally generate meaningful images

------------------------------------------------------------------------

## 🧠 Key Concepts Covered

-   Forward Diffusion Process (Adding Noise)
-   Reverse Diffusion Process (Denoising)
-   Noise Scheduling
-   Training Objective (MSE Loss)
-   Image Generation from Noise

------------------------------------------------------------------------

## 🏗️ Project Structure

    ├── model.py
    ├── diffusion.py
    ├── train.py
    ├── sample.py
    ├── utils.py
    ├── notebook.ipynb
    └── README.md

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python
-   PyTorch
-   NumPy
-   Matplotlib

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Clone the repository

    git clone https://github.com/Bushra-Abad/Denoising-diffusion-Model.git
    cd Denoising-diffusion-Model

### 2. Install dependencies

    pip install torch numpy matplotlib

### 3. Train the model

    python train.py

### 4. Generate samples

    python sample.py

------------------------------------------------------------------------

## 📊 Results

-   Model learns to progressively remove noise
-   Generates images from random Gaussian noise
-   Demonstrates the core idea behind modern generative AI models

------------------------------------------------------------------------

## 📖 Blog Explanation

For a detailed explanation, read the Medium article:

https://medium.com/@f223863/building-a-denoising-diffusion-probabilistic-model-ddpm-from-scratch-with-pytorch-ac4f07eaf363

------------------------------------------------------------------------

## 💡 Learning Outcome

-   Strong understanding of diffusion models
-   Hands-on experience with generative AI
-   Improved PyTorch skills

------------------------------------------------------------------------

## 🤝 Contributing

Feel free to fork and improve the project.

------------------------------------------------------------------------

## ⭐ Support

If you found this helpful, consider giving it a star!
