# Environment-Aware LocoMotion Generation using Autoregressive Diffusion Mdoels

This repository provides an implementation of an environment-aware motion generation framework based on conditioned autoregressive diffusion models.

The proposed method generates data-driven and controllable human motions conditioned on environmental constraints such as footstep information.

## 🔹 Overview

We propose an autoregressive diffusion-based motion generation framework that incorporates environmental awareness through footstep and trajectory constraints.

Unlike conventional diffusion-based motion models that are difficult to control in interactive scenarios, our approach integrates reinforcement learning and conditional guidance to enable real-time controllable motion synthesis.



## 🔹 Key Features

- Environment-aware motion generation
- Footstep-based conditioning
- Autoregressive diffusion architecture
- Reinforcement learning-based controllability
- Real-time inference (~30 FPS)


## 🔹 Method

Our framework consists of the following components:

1. Motion Encoder & Preprocessing
2. Autoregressive Diffusion Model
3. Footstep Conditioning
4. RL-based Control Module
5. Real-time Visualization System

### Autoregressive Diffusion

At each timestep, the model predicts future motion segments conditioned on past generated motion and environmental signals.

### Environment Conditioning

- Footstep contact constraints
- Trajectory guidance
- Obstacle-aware navigation

### RL-based Control

Reinforcement learning is used to improve controllability and responsiveness under user input and environmental changes.


## 🔹 Dataset

We use the following datasets:

- LAFAN1

### Preprocessing

- Motion mirroring for data augmentation
- Skeleton normalization
- Root alignment
- Velocity computation
- Foot contact extraction

All preprocessing scripts are provided in `dataset/`.


## 🔹 Installation

### Requirements

- Python 3.7+
- PyTorch 1.13+
- CUDA 11.x (optional, for GPU)
- NumPy, SciPy, Matplotlib
- For Rendering : Pybullet or Panda3D
- ...