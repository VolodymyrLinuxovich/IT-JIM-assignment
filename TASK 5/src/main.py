# src/main.py

import cv2
import numpy as np
import os

from detection.hand_landmarks_detector import HandLandmarksDetector
from pose_elimination.finger_pose_estimator import FingerPoseEstimator
from rendering.ring_renderer import RingRenderer

# Load camera intrinsics (example calibration values)
fx, fy, cx, cy = 800, 800, 320, 240
cam_mat = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float32)
dist = np.zeros((4, 1))

# Set project root directory (assumes this file is in src/)
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
img_dir = os.path.join(proj_root, 'data', 'trainee_program_2025_3D_task_data', 'images')
print("Images folder contents:", os.listdir(img_dir))

# Read input image (using a sample image, e.g. 'original_0.png')
img_path = os.path.join(img_dir, 'original_0.png')
img = cv2.imread(img_path)
if img is None:
    print("Could not open image!")
    exit(1)

# Detect hand landmarks using MediaPipe
detector = HandLandmarksDetector()
lms = detector.detect_landmarks(img)
if lms is None:
    print("Hand landmarks not found!")
    exit(1)

# Estimate the pose (translation and orientation) of the index finger
pose_est = FingerPoseEstimator()
f_pose = pose_est.estimate_pose(lms)
rvec, tvec = pose_est.get_pose_as_rvec_tvec(f_pose)

# Load the 3D ring model (make sure the file exists)
mdl_path = os.path.join(proj_root, 'data', 'trainee_program_2025_3D_task_data', 'models', 'ring', 'ring.obj')
ring_rend = RingRenderer(mdl_path, scale_factor=0.001)

# Render the ring onto the image by projecting 3D vertices onto the 2D image plane
res_img = ring_rend.render_ring(img, rvec, tvec, cam_mat, dist)

# Ensure output directory exists and save the resulting image
out_dir = os.path.join(proj_root, 'data', 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'result.jpg')
print("Saving result to:", out_path)
print("Result image type:", type(res_img))
print("Result image shape:", getattr(res_img, 'shape', 'None'))
cv2.imwrite(out_path, res_img)
