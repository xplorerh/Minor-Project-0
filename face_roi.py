import cv2
import mediapipe as mp
import numpy as np

from config import Config

# The FaceROIExtractor class performs three main tasks:

# Detects the face using MediaPipe Face Mesh.
# Creates a mask containing only facial skin.
# Computes the average Red, Green, and Blue (RGB) values from that skin region.

# Those RGB values are later passed to the POS algorithm to estimate heart rate.

class FaceROIExtractor:
    """
    Extracts facial ROI using MediaPipe Face Mesh.

    Creates a face mask from the full face mesh convex hull, then
    erodes it to exclude eyes, nose, and mouth — leaving mainly
    the forehead and cheeks for rPPG signal extraction.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_roi_mask(self, image, landmarks):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        points = np.array([
            (int(lm.x * w), int(lm.y * h)) for lm in landmarks
        ])

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        ksize = int(min(h, w) * Config.ROI_EROSION_RATIO)
        if ksize > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ksize, ksize)
            )
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def extract_rgb(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark
        mask = self.get_roi_mask(image, landmarks)

        if np.sum(mask) == 0:
            return None, mask

        roi = cv2.bitwise_and(image, image, mask=mask)
        mean_r = np.mean(roi[:, :, 2][mask > 0])
        mean_g = np.mean(roi[:, :, 1][mask > 0])
        mean_b = np.mean(roi[:, :, 0][mask > 0])

        return (mean_r, mean_g, mean_b), mask

    def visualize_roi(self, image, mask):
        overlay = image.copy()
        overlay[:, :, 2] = cv2.add(overlay[:, :, 2], mask)
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
