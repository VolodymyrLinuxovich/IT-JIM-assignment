# src/pose_estimation/finger_pose_estimator.py

import numpy as np
import cv2

class FingerPoseEstimator:
    def __init__(self):
        pass

    def estimate_pose(self, lm):
        """
        Estimate the 6DoF pose of the index finger using provided landmarks.
        Assumption: lm is a list of 21 points (x, y, z) from MediaPipe.
        Index finger keypoints: 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP).
        """
        # Select keypoints of the index finger:
        m = lm[5]  # MCP (metacarpophalangeal joint)
        p = lm[6]  # PIP (proximal interphalangeal joint)
        d = lm[7]  # DIP (distal interphalangeal joint)
        t = lm[8]  # TIP (fingertip)

        # Compute the normalized vector along the finger (from MCP to TIP)
        fv = t - m
        fv = fv / np.linalg.norm(fv)

        # To determine the full 3D orientation, including roll, we use an extra reference point.
        # Here we use landmark 17 (e.g., tip of the pinky) as a reference.
        ref = lm[17]
        rv = ref - m
        rv = rv / np.linalg.norm(rv)

        # Now, compute an orientation matrix based on the finger's vector fv.
        # z-axis: direction along the finger.
        z = fv
        # x-axis: projection of the reference vector onto the plane orthogonal to z.
        x = rv - np.dot(rv, z) * z
        x = x / np.linalg.norm(x)
        # y-axis: completes the right-handed coordinate system.
        y = np.cross(z, x)

        # The position (translation) for the ring is taken as the midpoint between MCP and PIP.
        cen = (m + p) / 2.0

        # Assemble the rotation matrix.
        rot_mat = np.column_stack((x, y, z))
        return {'center': cen, 'rotation_matrix': rot_mat}

    def get_pose_as_rvec_tvec(self, pose):
        """
        Convert the orientation matrix to a rotation vector (rvec) and obtain the translation vector (tvec)
        that are required for OpenCV's projectPoints.
        """
        R = pose['rotation_matrix']
        cen = pose['center']
        # Convert rotation matrix to rotation vector using Rodrigues formula.
        rvec, _ = cv2.Rodrigues(R)
        tvec = cen.reshape((3, 1))
        return rvec, tvec