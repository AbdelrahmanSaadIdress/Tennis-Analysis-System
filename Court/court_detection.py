class PositionExtractor:
    """
    PositionExtractor
    -----------------
    Converts bounding boxes into:
    - pixel positions (image space)
    - metric positions (meters)
    - mini-court positions

    Designed for fixed-court homography pipelines.
    """

    def __init__(self, court):
        """
        Args:
            court: CourtDrawing or similar object providing
                    image_to_meters() and image_to_mini_court()
        """
        self.court = court

    # ---------- Geometry ----------

    @staticmethod
    def calc_pos_from_bbox(bbox):
        """
        Calculate bottom-center position from bbox.

        Args:
            bbox (list or tuple): [x_min, y_min, x_max, y_max]

        Returns:
            (int, int): (x, y) bottom-center pixel position
        """
        x_min, y_min, x_max, y_max = bbox
        x = (x_min + x_max) / 2.0
        y = y_max
        return int(x), int(y)

    # ---------- Processing ----------

    def process_players(self, tracks, frame_id, frame_obj):
        for player_id, player_info in frame_obj.items():
            pixel_pos = self.calc_pos_from_bbox(player_info["bbox"])

            tracks["players"][frame_id][player_id].update({
                "player_position_in_pixels": pixel_pos,
                "player_position_in_meters": self.court.image_to_meters(pixel_pos),
                "player_position_in_minicourt": self.court.image_to_mini_court(pixel_pos),
            })

    def process_ball(self, tracks, track_id, frame_id, frame_obj):
        pixel_pos = self.calc_pos_from_bbox(frame_obj["bbox"])

        tracks[track_id][frame_id].update({
            "ball_position_in_pixels": pixel_pos,
            "ball_position_in_meters": self.court.image_to_meters(pixel_pos),
            "ball_position_in_minicourt": self.court.image_to_mini_court(pixel_pos),
        })

    
    def process_tracks(self, tracks):
        """
        Enrich tracks with position information.

        Modifies tracks in-place.
        """
        for track_id, track_obj in tracks.items():

            if track_id in {"persons", "court_keypoints"}:
                continue

            for frame_id, frame_obj in enumerate(track_obj):

                if frame_obj is None:
                    continue

                if track_id == "players":
                    self.process_players(tracks, frame_id, frame_obj)
                else:
                    self.process_ball(tracks, track_id, frame_id, frame_obj)


    def update_tracks(self, tracks, tranformed_tracks):

        for frame_id, frame_obj in enumerate(tracks["players"]):
            
            if frame_obj is None:
                continue

            for player_id, player_info in frame_obj.items():
                if player_id in tranformed_tracks[frame_id]:
                    tracks["players"][frame_id][player_id]["player_position_in_meters"] = tranformed_tracks[frame_id][player_id]["meters"]
        return tracks