# Edge-Detection-CNN-VGG16-and-Holistically-Nested-Edge-Detection

## Overview
This repository contains the implementation of edge detection using classical and deep learning methods. The goal is to detect edges that humans perceive as important for segmentation tasks, leveraging the **Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)**.

The notebook progresses through:

- Classical edge detection using the **Canny method**.
- A simple **CNN-based edge detection model**.
- A **VGG16-based model** with a transpose convolution decoder.
- A state-of-the-art **Holistically Nested Edge Detection (HED)** model.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tasks](#tasks)
  - [Task 1: Canny Edge Detection](#task-1-canny-edge-detection)
  - [Task 2: Simple CNN Model](#task-2-simple-cnn-model)
  - [Task 3: VGG16 Model](#task-3-vgg16-model)
  - [Task 4: Holistically Nested Edge Detection (HED)](#task-4-holistically-nested-edge-detection-hed)

## Dataset
The **Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)** is used for training, validation, and testing. The dataset contains images paired with human-annotated ground truth edge maps.

## Tasks

### Task 1: Canny Edge Detection
- Implemented classical **Canny edge detection**.
- Experimented with different **Gaussian blur sigma** values to study their effect on edge detection.
- Compared Canny outputs with ground truth annotations.

#### Canny Comparison
Below is an example comparison between the original image, ground truth edge map, and Canny output:

![Canny Comparison](canny_comparision.jpg)

### Task 2: Simple CNN Model
- Designed a simple 3-layer CNN with **ReLU activation** in hidden layers.
- Trained using **class-balanced loss** to address edge/non-edge imbalance.
- Compared test outputs with ground truth annotations after thresholding.

### Task 3: VGG16 Model
- Imported **VGG16** without the last max-pooling and fully connected layers.
- Added a **transpose convolution decoder** to restore output size to original image dimensions.
- Experimented with **bilinear interpolation upsampling** and compared results.

### Task 4: Holistically Nested Edge Detection (HED)
- Implemented a deep learning model inspired by the **HED** paper.
- Extracted side outputs from intermediate layers of **VGG16**.
- Upsampled side outputs using **bilinear interpolation** and fused them with learnable weights.
- Trained using **class-balanced loss** for 50 epochs.

