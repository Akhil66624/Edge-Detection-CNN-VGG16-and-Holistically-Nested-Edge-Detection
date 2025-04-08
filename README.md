# Edge-Detection-CNN-VGG16-and-Holistically-Nested-Edge-Detection

Overview
This repository contains the implementation of edge detection using classical and deep learning methods. The goal is to detect edges that humans perceive as important for segmentation tasks, leveraging the Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500).

The notebook progresses through:

Classical edge detection using the Canny method.

A simple CNN-based edge detection model.

A VGG16-based model with a transpose convolution decoder.

A state-of-the-art Holistically Nested Edge Detection (HED) model.

Table of Contents
Overview

Dataset

Tasks

Task 1: Canny Edge Detection

Task 2: Simple CNN Model

Task 3: VGG16 Model

Task 4: Holistically Nested Edge Detection (HED)

Installation

Usage

Results and Observations

Contributing

License

Dataset
The Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500) is used for training, validation, and testing. The dataset contains images paired with human-annotated ground truth edge maps.

Tasks
Task 1: Canny Edge Detection
Implemented classical Canny edge detection.

Experimented with different Gaussian blur sigma values to study their effect on edge detection.

Compared Canny outputs with ground truth annotations.

Key Question: Does the Canny detector work well for edge detection? Why/Why not?

![Canny Comparison](canny_comparisionTask 2: Simple CNN Model**

Designed a simple 3-layer CNN with ReLU activation in hidden layers.

Trained using class-balanced loss to address edge/non-edge imbalance.

Compared test outputs with ground truth annotations after thresholding.

Report Includes:

Description of the class-balanced loss function and why it is better than binary cross entropy loss.

Explanation of the activation function used (Sigmoid).

Training and validation loss curves.

Observations on model performance.

Task 3: VGG16 Model
Imported VGG16 without the last max-pooling and fully connected layers.

Added a transpose convolution decoder to restore output size to original image dimensions.

Experimented with bilinear interpolation upsampling and compared results.

Report Includes:

Description of the loss function and activation function used.

Training and validation loss curves.

Comparison of outputs between CNN and VGG16 models.

Task 4: Holistically Nested Edge Detection (HED)
Implemented a deep learning model inspired by HED paper.

Extracted side outputs from intermediate layers of VGG16.

Upsampled side outputs using bilinear interpolation and fused them with learnable weights.

Trained using class-balanced loss for 50 epochs.

Report Includes:

Training and validation loss curves.

Visualization of fused output and side outputs at multiple thresholds.

Learned weights for side outputs.

Observations on performance compared to CNN, VGG16, and Canny.

Installation
To set up the project locally:

Clone this repository:

bash
git clone <repository-url>
Install dependencies:

bash
pip install -r requirements.txt
Ensure OpenCV is installed:

bash
apt-get install -y libgl1-mesa-glx
Usage
Task 1: Canny Edge Detection
bash
python canny_edge_detection.py
Task 2: Simple CNN Model Training
bash
python train_simple_cnn.py
Task 3: VGG16 Model Training
bash
python train_vgg16.py
Task 4: HED Model Training
bash
python train_hed.py
Results and Observations
Task Comparisons:
Canny vs HED:

Canny detects all strong gradients indiscriminately, including irrelevant edges in textures like snow or ice.

HED focuses on perceptually significant edges by learning from human annotations.

CNN vs VGG16:

VGG16 outperforms the simple CNN due to its deeper architecture, capturing multi-scale features.

HED vs Other Models:

HED achieves the best performance by combining multi-scale features via learnable fusion weights.

Visualizations:
Plots include:

Original image, ground truth, fused output, and side outputs at multiple thresholds.

Contributing
Contributions are welcome! Please follow these steps:

Fork this repository.

Create a new branch (git checkout -b feature-name).

Commit your changes (git commit -m "Add feature").

Push to your branch (git push origin feature-name).

Open a pull request.

License
This project is licensed under the MIT License.

This README file follows GitHub's Markdown standards, providing clear instructions, descriptions, visualizations, and links for easy navigation through your repository!
