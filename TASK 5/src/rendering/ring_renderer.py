# src/rendering/ring_renderer.py

import cv2
import numpy as np
import pywavefront  # Used for loading .obj models

class RingRenderer:
    def __init__(self, mdl_path, scale=1.0):
        """
        mdl_path: path to the 3D ring model (.obj or .glb)
        scale: factor to scale the model (e.g., from millimeters to meters)
        """
        # Load the model using PyWavefront
        self.scn = pywavefront.Wavefront(mdl_path, create_materials=True, collect_faces=True)
        self.scale = scale
        
        # Assume the model stores its vertices as a list in self.scn.vertices.
        # Convert them to a NumPy array and apply scaling.
        self.verts = np.array(self.scn.vertices, dtype=np.float32) * self.scale

    def render_ring(self, img, rvec, tvec, cam_mat, dist=np.zeros((4,1))):
        """
        Renders the ring onto the image.
        Uses cv2.projectPoints to project the 3D vertices onto the 2D image plane.
        
        Parameters:
            img: Input image (as a NumPy array)
            rvec: Rotation vector (from cv2.Rodrigues)
            tvec: Translation vector (3x1)
            cam_mat: Camera intrinsic matrix
            dist: Distortion coefficients (default: zero distortion)
        
        Returns:
            A copy of the input image with the projected ring vertices drawn.
        """
        # Project 3D vertices into 2D image space using the provided camera parameters.
        pts, _ = cv2.projectPoints(self.verts, rvec, tvec, cam_mat, dist)
        pts = pts.reshape(-1, 2).astype(int)
        
        # Create a copy of the original image to draw on.
        out_img = img.copy()
        
        # For demonstration, draw a circle for each projected vertex.
        # In a full implementation, polygons (faces) could be filled with textures and shadows.
        for pt in pts:
            cv2.circle(out_img, tuple(pt), 2, (0, 255, 0), -1)
            
        return out_img
