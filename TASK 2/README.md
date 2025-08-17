# UAV Autonomous Navigation – Trajectory Reconstruction

## Overview

UAV Autonomous Navigation is a computer vision project that demonstrates how to reconstruct and visualize the flight path of an unmanned aerial vehicle (UAV) using a satellite panoramic image and a series of consecutive drone-captured frames. The system leverages feature matching and homography estimation to project the UAV’s positional data onto the global map, thereby enabling route reconstruction and visualization.

## Features

- **Global Map Loading:** Reads a high-resolution satellite panoramic image of the area of interest.
- **Frame Loading:** Loads a sequence of cropped images (frames) captured by the UAV during its flight.
- **Homography Computation:** Uses ORB feature detection and BFMatcher to compute homography between each frame and the global map.
- **Trajectory Reconstruction:** Calculates the transformed center point of each frame on the global map to form the UAV’s flight path.
- **Trajectory Smoothing:** Applies a simple moving average to smooth the raw trajectory data.
- **Visualization:** Displays the UAV trajectory on the global map with distinct markers for the start and finish.
- **Video Generation:** Optionally creates a video animation of the UAV’s route reconstruction with real-time trajectory progress.

## Project Structure

TASK 2/
├── data/
│   ├── global_map.png
│   └── crops/
├── README.md
├── requirements.txt
└── main.py

## Requirements

The project uses the following Python libraries:
- **OpenCV (opencv-python):** For image I/O, geometric transformations, homography computation, and video creation.
- **NumPy:** For numerical operations and array handling.
- **Matplotlib:** For plotting and visualization.
- **OS Module (Python Standard Library):** For file and directory operations.

See the `requirements.txt` file for the complete list of dependencies.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd project_root

2. **Install Dependencies:**
# Install all required packages using pip:
# 
# ```bash
# pip install -r requirements.txt
# ```

3. **Directory Setup:** 
# Ensure that the project directory is organized as described in the Project Structure section and that the input images are placed in their respective folders.

# Process Flow

# Input Loading:
# The tool reads a satellite panoramic image (global map) from the specified path.
# It loads a series of cropped frames (simulated UAV photos) from a designated directory.

# Feature Extraction & Homography Computation:
# For each frame, ORB features are detected and matched against the global map.
# A homography transformation is computed to project the center of each frame into the coordinate system of the global map.

# Trajectory Reconstruction:
# The centers of all frames are collected to form the raw UAV flight path.
# A moving average filter is applied to the raw points to create a smooth trajectory.

# Visualization & Video Generation:
# The final smoothed trajectory is visualized over the global map with clear markers indicating the start and finish.
# Optionally, a video animation is generated that shows the progression of the trajectory over time.

# Result Verification:
# After running the script, verify the output by:
# - Opening the visualization window to see the plotted trajectory overlaying the global map.
# - Checking for the generated video file (e.g., drone_route.avi) which animates the UAV’s flight path.

# Code Modules

# Global Map & Frame Loading:
# load_map:
# Loads and validates the global map image from disk.
# load_frms:
# Loads all cropped frames from the specified folder and sorts them in order.

# Homography and Center Point Calculation:
# find_homog:
# Computes the homography between a frame and the global map using ORB detector and BFMatcher.
# get_center:
# Transforms the center of each frame into the global map coordinate system using the computed homography.

# Trajectory Processing:
# smooth_traj:
# Applies a moving average filter to the raw trajectory points.
# vis_traj:
# Visualizes both raw and smoothed trajectories on the global map with appropriate markers and labels.

# Video Creation:
# create_vid:
# Generates a video that animates the UAV’s trajectory, highlighting the progression of the flight path.