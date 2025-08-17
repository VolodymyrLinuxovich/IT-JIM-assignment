# src/detection/hand_landmarks_detector.py

import cv2
import mediapipe as mp
import numpy as np

class HandLandmarksDetector:
    def __init__(self, mode=False, max_hands=1, det_conf=0.5, track_conf=0.5):
        # Initialize MediaPipe Hands with given parameters.
        self.mpH = mp.solutions.hands
        self.hands = self.mpH.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self.mpDr = mp.solutions.drawing_utils

    def detect_landmarks(self, img):
        """
        Detects hand landmarks in an input image.
        Returns a list of landmark points as [x, y, z] (in pixel coordinates for x,y).
        """
        # Convert image from BGR to RGB as required by MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        
        if res.multi_hand_landmarks:
            # Take the first detected hand
            handLM = res.multi_hand_landmarks[0]
            h, w, _ = img.shape
            # Create a list of points scaled to the image dimensions
            lms = []
            for lm in handLM.landmark:
                lms.append(np.array([lm.x * w, lm.y * h, lm.z]))
            return lms
        else:
            return None

    def draw_landmarks(self, img, handLM):
        """
        Draws hand landmarks and their connections on the image.
        """
        self.mpDr.draw_landmarks(img, handLM, self.mpH.HAND_CONNECTIONS)
        return img
