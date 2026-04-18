from .homography_transformer import HomographyTransformer

class CourtTrackTransformer:
    """
    Handles:
    - Homography transformation
    - Mapping player & ball positions
    - Converting to meters
    - Storing structured results
    """

    def __init__(self, keypoints, drawer):
        self.drawer = drawer

        # Define source points (court in image)
        self.src_pts = [
            keypoints[0],
            keypoints[1],
            keypoints[2],
            keypoints[3]
        ]

        # Define destination points (mini court)
        self.dst_pts = [
            (0, 0),
            (drawer.rect_width - 2 * drawer.offset, 0),
            (0, drawer.rect_height - 2 * drawer.offset),
            (
                drawer.rect_width - 2 * drawer.offset,
                drawer.rect_height - 2 * drawer.offset
            )
        ]

        # Initialize homography transformer
        self.transformer = HomographyTransformer(
            self.src_pts,
            self.dst_pts
        )

    def _map_point(self, sample_point):
        """
        Apply homography + clamp values.
        """
        mapped_point = self.transformer.map_point(sample_point)

        mapped_point = (
            mapped_point[0],
            mapped_point[1]
        )

        meters = self.drawer.convert_mini_court_position_to_meter(
            mapped_point[0],
            mapped_point[1]
        )

        return mapped_point, meters

    def transform_tracks(self, new_frames, tracks):
        """
        Main processing function.
        Returns transformed_tracks list.
        """

        transformed_tracks = []

        players_tracks = tracks.get("players", [])
        ball_tracks = tracks.get("ball", [])

        for idx, _ in enumerate(new_frames):

            frame_result = {}

            # ---------------- PLAYER 1 ----------------
            sample_point = players_tracks[idx].get(
                "player_1", {}
            ).get("player_position_in_pixels")

            if sample_point is not None:
                mapped_point, meters = self._map_point(sample_point)

                frame_result["player_1"] = {
                    "pixel": sample_point,
                    "mini_court": mapped_point,
                    "meters": meters
                }

            # ---------------- PLAYER 2 ----------------
            sample_point = players_tracks[idx].get(
                "player_2", {}
            ).get("player_position_in_pixels")

            if sample_point is not None:
                mapped_point, meters = self._map_point(sample_point)

                frame_result["player_2"] = {
                    "pixel": sample_point,
                    "mini_court": mapped_point,
                    "meters": meters
                }

            # ---------------- BALL ----------------
            sample_point = ball_tracks[idx].get(
                "ball_position_in_pixels"
            )

            if sample_point is not None:
                mapped_point, meters = self._map_point(sample_point)

                frame_result["ball"] = {
                    "pixel": sample_point,
                    "mini_court": mapped_point,
                    "meters": meters
                }

            transformed_tracks.append(frame_result)

        return transformed_tracks