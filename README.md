# Self-Driving Car Simulation - Model Training and Testing

## Overview

This project focuses on training a convlutional neural network (CNN) to steer a simulated self-driving car using image data and steering angles. The model learns from a dataset collected within a driving simulator and aims to generalize steering behavior in autonomous mode.

## Approach

1. **Data Loading & Preprocessing**:
   - Loaded driving image paths and steering angles from `driving_log.csv`.
   - Preprocessed images by cropping irrelevant areas, converting to YUV color space, applying Gaussian blur, resizing to (200x66), and normalizing pixel values.

2. **Data Augmentation**:
   - Applied transformations such as horizontal flipping, brightness adjustment, zoom, pan, and rotation to improve model generalization and reduce overfitting.

3. **Model Architecture**:
   - Followed the NVIDIA self-driving car architecture with 5 convolutional layers, followed by dense layers.
   - Used ReLU activations and the Adam optimizer with a small learning rate (`1e-4`) to stabilize training.

4. **Training**:
   - Used a generator function to batch and optionally augment data during training.
   - Trained the model for 10 epochs using Mean Squared Error (MSE) as the loss function and tracked Mean Absolute Error (MAE) as a metric.

5. **Testing**:
   - After training, the model was tested in the Unity simulator using `TestSimulation.py` to drive the car autonomously.

## Challenges Faced

- **Simulator not starting the car in autonomous mode**: This was resolved by ensuring the Unity simulator was opened and set to Autonomous Mode before launching `TestSimulation.py`.
- **Model performance**: The trained model tended to drive on the edge of the road rather than the center. This may be due to imbalanced training data where off-center positions were overrepresented or the lack of diverse scenarios in training.

## How to Run

**Environment Setup**

1. Make sure Anaconda is installed.
2. Run the following commands to set up dependencies:

`conda create --name driving_sim --file package_list.txt`
`conda activate driving_sim`

**Data Collection**
1. Open the Unity Driving Simulator and manually drive a car to collect data.
2. The simulator will generate a driving_log.csv file and a folder with image frames (IMG/).

**Training**

Run the training script:
`python driving_train.py`

This will:
- Preprocess and augment the collected data
- Train the CNN model
- Save the trained model as model.h5
- Plot training/validation loss and MAE

**Testing**

Open the Unity simulator and switch to Autonomous Mode and run:
`python TestSimulation.py`

Results:
- The model was able to follow the road but exhibited a tendency to drift toward the borders.
- Training and validation loss graphs showed consistent convergence without significant overfitting.
