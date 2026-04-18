import cv2

class TacticalViewAnnotator:
    """
    Responsible only for drawing annotations.
    Does NOT compute homography.
    """

    def __init__(self, drawer):
        """
        drawer: your mini court drawer object
        """
        self.drawer = drawer

    def draw_point(self, frame, pixel_point, mapped_point,
                   color=(210, 78, 255), radius=12):

        # Draw original pixel point
        cv2.circle(
            frame,
            (int(pixel_point[0]), int(pixel_point[1])),
            radius,
            color,
            -1
        )

        # Draw mini court mapped point
        cv2.circle(
            frame,
            (
                int(self.drawer.inner_x1 + mapped_point[0]),
                int(self.drawer.inner_y1 + mapped_point[1])
            ),
            radius,
            color,
            -1
        )

    def draw_frame_annotations(self, frame, frame_result):
        """
        frame_result = {
            'player_1': {...},
            'player_2': {...},
            'ball': {...}
        }
        """

        if "player_1" in frame_result:
            self.draw_point(
                frame,
                frame_result["player_1"]["pixel"],
                frame_result["player_1"]["mini_court"],
                color=(0, 255, 0)
            )

        if "player_2" in frame_result:
            self.draw_point(
                frame,
                frame_result["player_2"]["pixel"],
                frame_result["player_2"]["mini_court"],
                color=(255, 0, 0)
            )

        if "ball" in frame_result:
            self.draw_point(
                frame,
                frame_result["ball"]["pixel"],
                frame_result["ball"]["mini_court"],
                color=(0, 165, 255),
                radius=8
            )

        return frame