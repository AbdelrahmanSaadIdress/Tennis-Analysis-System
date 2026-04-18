import cv2
import numpy as np


class GrayRectangleDrawer:
    """
    Draws a filled background rectangle with an inner rectangle.
    Position is fixed relative to the frame using margins.
    """

    def __init__(
        self,
        rect_width=100,
        rect_height=100,
        right_margin=20,
        top_margin=20,
        back_color=(109, 220, 33),   # BGR
        for_color=(128, 88, 70),    # BGR
        offset=20
    ):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.right_margin = right_margin
        self.top_margin = top_margin

        self.back_color = back_color
        self.for_color = for_color
        self.offset = offset

        self._outer_pos = None  # (x, y)

    # ===============================
    # Private helpers
    # ===============================

    def _compute_outer_position(self, frame_shape):
        """
        Compute and cache the outer rectangle top-left position.
        """
        H, W = frame_shape[:2]

        if self._outer_pos is None:
            x = W - self.rect_width - self.right_margin
            y = self.top_margin
            self._outer_pos = (x, y)

        return self._outer_pos

    def _draw_outer_rectangle(self, frame, x, y):
        """
        Draw filled outer rectangle.
        """
        x2 = x + self.rect_width
        y2 = y + self.rect_height

        cv2.rectangle(
            frame,
            (x, y),
            (x2, y2),
            self.back_color,
            -1
        )

        return x2, y2

    def _draw_inner_rectangle(self, frame, x, y, x2, y2):
        """
        Draw filled inner rectangle using offset.
        """
        self.inner_x1 = x + self.offset
        self.inner_y1 = y + self.offset
        inner_x2 = x2 - self.offset
        inner_y2 = y2 - self.offset

        cv2.rectangle(
            frame,
            (self.inner_x1, self.inner_y1),
            (inner_x2, inner_y2),
            self.for_color,
            -1
        )

        return (self.inner_x1, self.inner_y1, inner_x2, inner_y2)

    def get_keypoints_in_meters(self):
        return[
            (0,0),
            (10.97, 0), 
            (0, 23.77),
            (10.97, 23.77),

            (1.37,0),
            (1.37, 23.77),
            (10.97-1.37, 0),
            (10.97-1.37, 23.77),

            (1.37, 5.48),
            (10.97-1.37, 5.48),
            (1.37,23.77-5.48),
            (10.97-1.37, 23.77-5.48),

            (10.97/2, 5.48),
            (10.97/2, 23.77-5.48)
        ]

    def convert_to_minicourt_coordinates(self, x, y, court_width_m=10.97, court_height_m=23.77):
        scale_x =  (self.rect_width - 2*self.offset) / court_width_m
        scale_y = (self.rect_height - 2*self.offset) / court_height_m 
        return x * scale_x, y * scale_y

    def draw_keypoints_on_minicourt(self, frame):
        self.keypoints = self.get_keypoints_in_meters()

        for i, (x, y) in enumerate(self.keypoints):
            cv2.circle(
                frame,
                (int(self.inner_x1+self.convert_to_minicourt_coordinates(x, y)[0]), int(self.inner_y1+self.convert_to_minicourt_coordinates(x, y)[1])),
                radius=7,
                color=(0, 0, 0),  # Blue color
                thickness=-1
            )
            cv2.putText(
                frame,
                f"K{i}",
                (int(self.inner_x1+self.convert_to_minicourt_coordinates(x, y)[0]) + 8, int(self.inner_y1+self.convert_to_minicourt_coordinates(x, y)[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

    def _keypoint_to_pixel(self, idx):
        """
        Convert keypoint at index idx to pixel coordinate in frame.
        """
        x_m, y_m = self.keypoints[idx]
        x_px, y_px = self.convert_to_minicourt_coordinates(x_m, y_m)
        return int(self.inner_x1 + x_px), int(self.inner_y1 + y_px)
    
    def get_keypoints_in_pixels(self):
        return [
            self._keypoint_to_pixel(i) for i in range(len(self.keypoints))
        ]

    def draw_line_between_2_points_indices(self, frame, idx1, idx2, color=(255,255,255), thickness=2):
        """
        Draw a line between two keypoints using their indices.
        """
        pt1 = self._keypoint_to_pixel(idx1)
        pt2 = self._keypoint_to_pixel(idx2)
        cv2.line(frame, pt1, pt2, color, thickness)

    def draw_lines_on_minicourt(self, frame):
        """
        Draw multiple lines on the frame by specifying the indices to connect.
        """
        # Example: connect points 0→1, 1→2, 2→3
        lines_to_draw = [
            (0, 1),
            (1, 3),
            (0, 2),
            (2, 3),

            (4,5),
            (6,7),

            (8,9),
            (10,11),

            (12,13)
        ]

        for idx1, idx2 in lines_to_draw:
            self.draw_line_between_2_points_indices(frame, idx1, idx2, color=(255,255,255), thickness=2)

        return frame

    # ===============================
    # Public method     
    # ===============================

    def draw(self, frames):
        """
        Main drawing function.
        """
        output_frames = []
        self.keypoints = self.get_keypoints_in_meters()
        
        for frame in frames:
            frame_copy = frame.copy()
            # Compute outer position
            x, y = self._compute_outer_position(frame_copy.shape)

            # Draw outer
            x2, y2 = self._draw_outer_rectangle(frame_copy, x, y)

            # Draw inner
            self._draw_inner_rectangle(frame_copy, x, y, x2, y2)

            self.draw_lines_on_minicourt(frame_copy)
            self.draw_keypoints_on_minicourt(frame_copy)


            output_frames.append(frame_copy)


        self._print_debug_info()
        return output_frames

    # ===============================
    # Debug
    # ===============================
    def _print_debug_info(self):
        x, y = self._outer_pos
        print(f"Outer: x={x}, y={y}, w={self.rect_width}, h={self.rect_height}")
        print(f"Inner offset: {self.offset}")

    def convert_mini_court_position_to_meter(self, x, y):
        """
        Convert mini court pixel coordinates to real-world meter coordinates.
        """
        # Assuming the mini court is scaled to fit within rect_width and rect_height
        # and that the real tennis court is 10.97m wide and 23.77m tall
        court_width_m = 10.97
        court_height_m = 23.77

        # Compute scaling factors
        scale_x = court_width_m / (self.rect_width - 2 * self.offset)
        scale_y = court_height_m / (self.rect_height - 2 * self.offset)

        # Convert pixel coordinates to meters
        x_m = x * scale_x
        y_m = y * scale_y

        return x_m, y_m
