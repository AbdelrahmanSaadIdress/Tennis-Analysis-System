import cv2
import numpy as np


class CourtDrawing:
    """
    CourtDrawing
    -------------
    Converts detected court keypoints into:
    - Metric court coordinates (meters)
    - Mini-court visualization
    """

    # ------------------ Court dimensions (meters) ------------------
    COURT_W = 10.97
    COURT_H = 23.77

    MARGIN_W = 0.91
    MARGIN_H = 6.40

    FULL_W = COURT_W + 2 * MARGIN_W
    FULL_H = COURT_H + 2 * MARGIN_H

    SERVICE_DIST = 6.40
    SIDELINE_DIST = 1.37

    # ---------------------------------------------------------------

    def __init__(self, frame, raw_keypoints, model_input_size=224):
        """
        Parameters
        ----------
        frame : np.ndarray
            Original video frame (H,W,3)
        raw_keypoints : np.ndarray
            Model-predicted keypoints in model-input space
        model_input_size : int
            Size used during model inference (default 224)
        """

        self.frame = frame
        self.h, self.w = frame.shape[:2]

        self.keypoints = self._scale_keypoints(
            raw_keypoints, model_input_size
        )

        self.outer_pts = self.keypoints[:4]
        self.delta_pts = self._compute_delta_corners()

        self.H_img2m = self._compute_img_to_meters_homography()
        self.H_img2mini = None  # computed lazily

    # ===================== Keypoint handling =====================

    def _scale_keypoints(self, keypoints, model_size):
        sx = self.w / model_size
        sy = self.h / model_size
        return np.array(
            [[x * sx, y * sy] for x, y in keypoints],
            dtype=np.float32
        )

    # ===================== Geometry ==============================

    def _compute_delta_corners(self):
        """
        Computes inner (margin-expanded) rectangle in image space.
        """
        k = self.outer_pts

        px_w1 = abs(k[0][0] - k[1][0])
        px_w2 = abs(k[2][0] - k[3][0])

        px_h1 = abs(k[0][1] - k[2][1])
        px_h2 = abs(k[1][1] - k[3][1])

        dx1 = (self.MARGIN_W * px_w1) / self.COURT_W
        dx2 = (self.MARGIN_W * px_w2) / self.COURT_W

        dy1 = (self.MARGIN_H * px_h1) / self.COURT_H
        dy2 = (self.MARGIN_H * px_h2) / self.COURT_H

        p0 = (k[0][0] - dx1, k[0][1] - dy1)  # TL
        p1 = (k[1][0] + dx2, k[1][1] - dy1)  # TR
        p2 = (k[2][0] - dx1, k[2][1] + dy2)  # BL
        p3 = (k[3][0] + dx2, k[3][1] + dy2)  # BR

        return np.array([p0, p1, p2, p3], dtype=np.float32)

    # ===================== Homographies ===========================

    def _compute_img_to_meters_homography(self):
        court_m = np.array([
            [0.0, 0.0],
            [self.FULL_W, 0.0],
            [0.0, self.FULL_H],
            [self.FULL_W, self.FULL_H],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(self.delta_pts, court_m)
        return H

    # ===================== Mapping ================================

    def image_to_meters(self, pt_img):
        pt = np.array([[pt_img]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H_img2m)
        return tuple(mapped[0, 0])

    # ===================== Mini court =============================

    def _create_mini_mapper(self, court_height_px=700, margin=30):
        full_h_px = court_height_px
        full_w_px = int(full_h_px * (self.FULL_W / self.FULL_H))

        x0 = self.w - full_w_px - margin
        y0 = margin

        sx = full_w_px / self.FULL_W
        sy = full_h_px / self.FULL_H

        def meters_to_px(x, y):
            return int(x0 + x * sx), int(y0 + y * sy)

        return meters_to_px, (x0, y0, full_w_px, full_h_px)

    def image_to_mini_court(self, pt_img):
        if self.H_img2mini is None:
            self._build_img_to_mini_homography()

        pt = np.array([[pt_img]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H_img2mini)
        return tuple(mapped[0, 0].astype(int))

    def _build_img_to_mini_homography(self):
        P, _ = self._create_mini_mapper()

        mini_corners = np.array([
            P(0, 0),
            P(self.FULL_W, 0),
            P(0, self.FULL_H),
            P(self.FULL_W, self.FULL_H),
        ], dtype=np.float32)

        self.H_img2mini, _ = cv2.findHomography(
            self.outer_pts, mini_corners
        )

    # ===================== Drawing ================================

    def draw_mini_court(self, frame, court_height_px=700, margin_right=30, margin_top=40):
        """
        Draws a metric-accurate mini court with margins on top of the frame.
        All geometry is done in METERS, then projected to pixels.
        """

        # ---- create mini-court mapper ----
        P, (x0, y0, full_w_px, full_h_px) = self._create_mini_mapper(
            court_height_px=court_height_px,
            margin=margin_right
        )

        x1 = x0 + full_w_px
        y1 = y0 + full_h_px

        # ---- work on a copy (CRITICAL) ----
        overlay = frame.copy()

        # ---- background (green) ----
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            (40, 135, 40),
            -1
        )

        alpha = 1.0
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # ---- drawing parameters ----
        white = (255, 255, 255)
        lw = 2

        # ---- court rectangle in METERS ----
        court_x0 = self.MARGIN_W
        court_y0 = self.MARGIN_H - 6
        court_x1 = self.MARGIN_W + self.COURT_W
        court_y1 = self.MARGIN_H + self.COURT_H + 6

        # ---- outer singles court ----
        cv2.rectangle(
            frame,
            P(court_x0, court_y0),
            P(court_x1, court_y1),
            white,
            lw
        )

        # ---- singles sidelines ----
        SIDELINE = self.SIDELINE_DIST
        cv2.line(
            frame,
            P(court_x0 + SIDELINE, court_y0),
            P(court_x0 + SIDELINE, court_y1),
            white,
            lw
        )
        cv2.line(
            frame,
            P(court_x1 - SIDELINE, court_y0),
            P(court_x1 - SIDELINE, court_y1),
            white,
            lw
        )

        # ---- net ----
        cv2.line(
            frame,
            P(court_x0, 6 + court_y0 + self.COURT_H / 2),
            P(court_x1, 6 + court_y0 + self.COURT_H / 2),
            white,
            lw
        )

        # ---- service lines ----
        SERVICE = self.SERVICE_DIST
        cv2.line(
            frame,
            P(court_x0 + SIDELINE, court_y0 + SERVICE),
            P(court_x1 - SIDELINE, court_y0 + SERVICE),
            white,
            lw
        )
        cv2.line(
            frame,
            P(court_x0 + SIDELINE, court_y1 - SERVICE),
            P(court_x1 - SIDELINE, court_y1 - SERVICE),
            white,
            lw
        )

        # ---- center service line ----
        cv2.line(
            frame,
            P(court_x0 + self.COURT_W / 2, court_y0 + SERVICE),
            P(court_x0 + self.COURT_W / 2, court_y1 - SERVICE),
            white,
            lw
        )

        return frame

    def draw_circle_on_minicourt(self, frame, x_meters, y_meters, color=(0, 0, 255), radius=5):
        """
        Draws a circle representing a position in the mini court overlay.
        x_meters, y_meters: position in court meters coordinates
        """
        # Lazily create the mini-court mapper
        if not hasattr(self, "_mini_mapper"):
            self._mini_mapper, _ = self._create_mini_mapper()

        # Convert meters to mini-court pixel coordinates
        px, py = self._mini_mapper(x_meters, y_meters)

        cv2.circle(frame, (px, py), radius, color, -1)


