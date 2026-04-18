import cv2
import numpy as np

class HomographyTransformer:
    """
    Homography transformer for a fixed camera.
    Computes the homography once and reuses it for all frames.
    """
    def __init__(self, src_points, dst_points, dst_size=None):
        """
        Initialize the transformer.

        Parameters:
            src_points (np.array): Nx2 array of points in the source image.
            dst_points (np.array): Nx2 array of corresponding points in the destination.
            dst_size (tuple): (width, height) of the output image. If None, will be determined from the first frame.
        """
        self.src_pts = np.array(src_points, dtype=np.float32)
        self.dst_pts = np.array(dst_points, dtype=np.float32)
        self.dst_size = dst_size
        self.H = None  # Homography matrix

    def compute_homography(self):
        """Compute homography matrix once."""
        self.H, status = cv2.findHomography(self.src_pts, self.dst_pts, cv2.RANSAC)
        if self.H is None:
            raise ValueError("Homography computation failed.")
        return self.H

    def apply(self, frame):
        """Apply the precomputed homography to a frame."""
        if self.H is None:
            self.compute_homography()
        if self.dst_size is None:
            h, w = frame.shape[:2]
            self.dst_size = (w, h)
        warped_frame = cv2.warpPerspective(frame, self.H, self.dst_size)
        return warped_frame

    def map_point(self, point):
        """
        Map a single point (x, y) from source to destination using the homography.
        """
        if self.H is None:
            self.compute_homography()
        pts = np.array([[point]], dtype=np.float32)  # shape (1,1,2)
        dst_pts = cv2.perspectiveTransform(pts, self.H)
        x_mapped, y_mapped = dst_pts[0, 0]
        return (x_mapped, y_mapped)
