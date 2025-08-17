# Virtual Ring Try-On

## Overview

Virtual Ring Try-On is a computer vision and 3D augmentation project that simulates how a ring looks on a specific finger from a given hand image. The system uses MediaPipe for hand landmarks detection, performs 6DoF pose estimation of the index finger, and renders a 3D ring model onto the hand image through projection using OpenCV. This project is designed for offline processing using provided images, 3D model data, and ARKit logs (including depth maps and canonical measurements).

## Features

- **Hand Landmark Detection:** Uses MediaPipe to extract 21 hand landmarks from input images.
- **6DoF Pose Estimation:** Estimates the translation and orientation of the index finger (using keypoints from the finger's MCP, PIP, DIP, and TIP) for accurate ring positioning.
- **3D Ring Projection:** Loads a 3D ring model (e.g., in OBJ format) using PyWavefront and projects its vertices onto the 2D image plane based on the computed pose.
- **Simple Visualization:** Draws the projected vertices for demonstration purposes. (The renderer can be enhanced to fill polygons, add textures, shadows, and occlusion effects.)
- **Modular Design:** The code is organized into modules for detection, pose estimation, and rendering.

## Project Structure

project_root/
├── data/
│   ├── trainee_program_2025_3D_task_data/
│   │   ├── images/
│   │   │   ├── original_0.png
│   │   │   ├── original_1.png
│   │   │   ├── original_2.png
│   │   │   ├── original_3.png
│   │   │   ├── original_9.png
│   │   │   ├── depth_0.png
│   │   │   ├── depth_1.png
│   │   │   ├── depth_2.png
│   │   │   ├── depth_3.png
│   │   │   └── depth_9.png
│   │   ├── models/
│   │   │   └── ring/
│   │   │       └── ring.obj
│   └── outputs/
│       └── result.jpg  (generated output)
├── src/
│   ├── detection/
│   │   └── hand_landmarks_detector.py
│   ├── pose_estimation/
│   │   └── finger_pose_estimator.py
│   ├── rendering/
│   │   └── ring_renderer.py
│   └── main.py
├── README.md
└── requirements.txt

## Requirements

The project uses the following Python libraries:

- **OpenCV (opencv-python):** For image I/O, geometric transformations, and projecting 3D points.
- **NumPy:** For numerical operations and handling matrices.
- **MediaPipe:** To detect hand landmarks efficiently.
- **PyWavefront:** To load and parse 3D models in OBJ format.

See the [requirements.txt] file for the exact list of dependencies.

## Installation

Follow these steps to set up your development environment:

# 1. Clone the Repository:
# 
# ```bash
# git clone <repository_url>
# cd project_root
# ```
# 
# 2. Install the Required Dependencies:
# 
# Use pip to install everything listed in `requirements.txt`:
# 
# ```bash
# pip install -r requirements.txt
# ```
# 
# This command will install OpenCV, NumPy, MediaPipe, and PyWavefront.
# 
# 3. Directory Structure:
# 
# Ensure the project directory is organized as described in the "Project Structure" section,
# with input images and 3D model files in the correct folders.


# Process Flow

**Camera Intrinsics:**  
The camera intrinsics are loaded and defined.

**Project Root Calculation:**  
The project root is calculated so that all relative paths work correctly.

**Image Loading:**  
An input hand image (e.g., `original_0.png`) is loaded.

**Hand Landmark Detection:**  
Hand landmarks are detected from this image using MediaPipe.

**Pose Computation:**  
The pose (rotation and translation) of the index finger is computed from the detected landmarks.

**3D Ring Model Loading:**  
A 3D ring model is loaded using PyWavefront. Its vertices are scaled as needed.

**3D to 2D Projection:**  
OpenCV’s `projectPoints` function is used to project the 3D vertices onto the 2D image plane, placing the ring in the proper position and orientation.

**Output Saving:**  
The output image (with the ring overlay) is saved to `data/outputs/result.jpg`.

# Result Verification

After running the script, check the `data/outputs` folder for the generated image. The image should display the original hand with the ring rendered on the index finger.

# Code Modules

## Hand Landmarks Detection

**File:** `src/detection/hand_landmarks_detector.py`

**Function:**

- Converts the BGR image to RGB and processes it with MediaPipe to extract 21 hand landmarks.
- Returns a list of 3D points (x, y in pixel coordinates and z as a relative depth) scaled to the image dimensions.

## Finger Pose Estimation

**File:** `src/pose_estimation/finger_pose_estimator.py`

**Function:**

- Uses specific landmarks from the index finger (MCP, PIP, DIP, TIP) to compute a normalized finger vector.
- Calculates roll by referencing an additional landmark (pinky tip).
- Creates an orientation matrix (with x, y, and z axes) and computes a center point between MCP and PIP to define the translation.
- Converts the orientation matrix into a rotation vector (using Rodrigues) and a translation vector for 3D projection.

## Ring Rendering

**File:** `src/rendering/ring_renderer.py`

**Function:**

- Loads the 3D ring model with PyWavefront, scales its vertices, and projects them onto the input image using the computed rotation and translation vectors.
- For demonstration, the projected vertices are drawn as small circles on a copy of the original image.

## Main Script

**File:** `src/main.py`

**Function:**

- Sets up camera parameters, loads the input image, and determines file paths using the project root.
- Calls the detection, pose estimation, and rendering modules sequentially.
- Saves the final augmented image to the outputs folder.
- This script is executed immediately without wrapping inside a function, as per user preference.

# Advanced Features & Future Work

## Improved Rendering

- Implement full polygon rendering with textures, shading, reflections, and occlusion using OpenGL, Blender's Python API, or a dedicated 3D engine.

## Depth-based Occlusion

- Integrate depth map data from ARKit logs to correctly handle occlusion (e.g., hiding parts of the ring that are behind the hand).

## Realistic Lighting

- Add realistic lighting effects by using HDR images (environment maps) or physically based rendering (PBR) shaders.

## Batch Processing & Animation

- Extend the project to process multiple images or video sequences and generate animations or GIFs of the ring try-on.

# Troubleshooting

## File Path Issues

- Ensure the project directory structure is maintained and all file paths referenced in the code match your local setup.

## Hand Landmark Detection Failures

- Make sure to use a clear, colored image of the hand (e.g., `original_0.png`) rather than a depth map, as MediaPipe requires sufficient contrast and color information.

## 3D Model Compatibility

- If PyWavefront displays warnings about unsupported OBJ format statements (such as smoothing groups or line segments), consider pre-processing the OBJ file using Blender or another tool to remove these unsupported elements.