import cv2
from utils import save_video
import numpy as np
import pandas as pd
from Court import CourtDrawing

class FramesAnnotations:
    def __init__(self, tracks, frames):
        """
        tracks: dict containing persons, players, ball, and court_keypoints
        """
        self.tracks = tracks
        court_kpts = tracks.get("court_keypoints")
        if court_kpts is not None:
            self.court = CourtDrawing(
                frames[0],
                court_kpts,
                model_input_size=224
            )

    def draw(self, frames, df):
        self.df = df

        # frames = self.draw_persons(frames)
        frames = self.draw_players(frames)      
        frames = self.draw_ball(frames)
        frames = self.draw_court_keypoints(frames)
        # frames = self.draw_mini_court(frames)
        frames = self.draw_frame_id(frames)
        frames = self.draw_analysis_table(frames)
        # frames = self.draw_minicourt_positions(frames)

        save_video(frames, "Outputs/Output_Videos/result.mp4")
        return frames

    def _get_df_row(self, frame_idx):
        if frame_idx >= len(self.df):
            return None
        return self.df.iloc[frame_idx]

    # --------------------------------------------------
    # Draw persons (bounding boxes + IDs)
    # --------------------------------------------------
    def draw_persons(self, frames):
        persons_tracks = self.tracks.get("persons", [])

        for frame_idx, frame in enumerate(frames):
            if frame_idx >= len(persons_tracks):
                continue

            frame_persons = persons_tracks[frame_idx]

            for person_id, person_data in frame_persons.items():
                x1, y1, x2, y2 = map(int, person_data["bbox"])

                # Bounding box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                            color=(0, 255, 0), thickness=2)

                # Label
                cv2.putText(
                    frame,
                    f"ID {person_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return frames

    # --------------------------------------------------
    # Draw players (bounding boxes + IDs) in blue
    # --------------------------------------------------
    def draw_players(self, frames):
        players_tracks = self.tracks.get("players", [])

        for frame_idx, frame in enumerate(frames):
            if frame_idx >= len(players_tracks):
                continue

            frame_players = players_tracks[frame_idx]

            for player_id, pdata in frame_players.items():
                x1, y1, x2, y2 = map(int, pdata["bbox"])

                # Bounding box (blue)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                            color=(255, 0, 0), thickness=2)

                # Label
                cv2.putText(
                    frame,
                    f"Player {player_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        return frames

    # --------------------------------------------------
    # Draw ball (bounding box)
    # --------------------------------------------------
    def draw_ball(self, frames):
        ball_tracks = self.tracks.get("ball", [])

        for frame_idx, frame in enumerate(frames):
            if frame_idx >= len(ball_tracks):
                continue

            ball_data = ball_tracks[frame_idx]

            if not ball_data or "bbox" not in ball_data:
                continue

            x1, y1, x2, y2 = map(int, ball_data["bbox"])

            # Bounding box (red)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                        color=(0, 0, 255), thickness=2)

            # Label
            cv2.putText(
                frame,
                "Ball",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return frames

    # --------------------------------------------------
    # Draw court keypoints (same on all frames)
    # --------------------------------------------------
    def draw_court_keypoints(self, frames):
        keypoints = self.tracks.get("court_keypoints", None)

        if keypoints is None:
            return frames

        MODEL_IMG_SIZE = 224  # keypoints are predicted in 224x224 space

        for frame in frames:
            H, W = frame.shape[:2]
            scale_x = W / MODEL_IMG_SIZE
            scale_y = H / MODEL_IMG_SIZE

            for (x, y) in keypoints:
                px = int(x * scale_x)
                py = int(y * scale_y)

                cv2.circle(
                    frame,
                    (px, py),
                    radius=6,
                    color=(0, 0, 0),   # BLACK (BGR)
                    thickness=-1
                )

        return frames

    # --------------------------------------------------
    # Draw frame ID on the top-right corner
    # --------------------------------------------------
    def draw_frame_id(self, frames):
        for frame_idx, frame in enumerate(frames):
            H, W = frame.shape[:2]
            text = f"Frame {frame_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness = 2

            # Get text size to position it at top-right
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
            org = (W - text_width - 10, 30)  # 10px from right, 30px from top

            cv2.putText(
                frame,
                text,
                org,
                font,
                scale,
                (0, 255, 255),  # Yellow color
                thickness
            )

        return frames

    def draw_analysis_table(self, frames):
        players_tracks = self.tracks.get("players", [])

        last = {
            "p1_shot": None, "p2_shot": None,
            "p1_avg_shot": None, "p2_avg_shot": None,
            "p1_move": None, "p2_move": None,
            "p1_avg_move": None, "p2_avg_move": None,
        }

        for frame_idx, frame in enumerate(frames):

            df_row = self._get_df_row(frame_idx)

            # -------------------------------
            # BALL STATS (persist values)
            # -------------------------------
            if df_row is not None:
                hitter = df_row.get("hitter")
                shot = df_row.get("shot_speed_kmh")
                avg = df_row.get("avg_shot_speed_kmh")

                if hitter == "player_1" and not pd.isna(shot):
                    last["p1_shot"] = shot
                    last["p1_avg_shot"] = avg
                if hitter == "player_2" and not pd.isna(shot):
                    last["p2_shot"] = shot
                    last["p2_avg_shot"] = avg

            # -------------------------------
            # PLAYER MOVEMENT STATS
            # -------------------------------
            if frame_idx < len(players_tracks):
                frame_players = players_tracks[frame_idx]
                if "player_1" in frame_players:
                    s = frame_players["player_1"].get("speed")
                    a = frame_players["player_1"].get("avg_speed")
                    if s is not None:
                        last["p1_move"] = s
                    if a is not None:
                        last["p1_avg_move"] = a

                if "player_2" in frame_players:
                    s = frame_players["player_2"].get("speed")
                    a = frame_players["player_2"].get("avg_speed")
                    if s is not None:
                        last["p2_move"] = s
                    if a is not None:
                        last["p2_avg_move"] = a

            # -------------------------------
            # Draw table
            # -------------------------------
            x, y = 20, 50
            width, height = 520, 160

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thickness = 1
            line_h = 22

            def fmt(v):
                return "--" if v is None or np.isnan(v) else f"{v:.1f}"

            lines = [
                "ANALYSIS SUMMARY",
                "-----------------------------",
                f"P1 Shot Speed     : {fmt(last['p1_shot'])} km/h",
                f"P1 Avg Shot Speed : {fmt(last['p1_avg_shot'])} km/h",
                f"P1 Move Speed     : {fmt(last['p1_move'])} km/h",
                f"P1 Avg Move Speed : {fmt(last['p1_avg_move'])} km/h",
                "",
                f"P2 Shot Speed     : {fmt(last['p2_shot'])} km/h",
                f"P2 Avg Shot Speed : {fmt(last['p2_avg_shot'])} km/h",
                f"P2 Move Speed     : {fmt(last['p2_move'])} km/h",
                f"P2 Avg Move Speed : {fmt(last['p2_avg_move'])} km/h",
            ]

            for i, text in enumerate(lines):
                cv2.putText(frame, text, (x + 12, y + 28 + i * line_h),
                            font, scale, (255, 255, 255), thickness)

        return frames

    

    def draw_mini_court(self, frames):
        if self.court is None:
            return frames

        for i, frame in enumerate(frames):
            frames[i] = self.court.draw_mini_court(frame)

        return frames


    def draw_minicourt_positions(self, frames):
        if self.court is None:
            return frames

        players_tracks = self.tracks.get("players", [])
        ball_tracks = self.tracks.get("ball", [])

        for frame_idx, frame in enumerate(frames):
            # Draw players
            if frame_idx < len(players_tracks):
                frame_players = players_tracks[frame_idx]
                for player_id, pdata in frame_players.items():
                    pos = pdata.get("player_position_in_minicourt")
                    if pos is not None:
                        x, y = pos
                        frame = self.court.draw_circle_on_minicourt(
                            frame, x, y, color=(0, 0, 255), radius=5
                        )

            # Draw ball
            if frame_idx < len(ball_tracks):
                ball_data = ball_tracks[frame_idx]
                if ball_data is not None:
                    pos = ball_data.get("ball_position_in_minicourt")
                    if pos is not None:
                        x, y = pos
                        frame = self.court.draw_circle_on_minicourt(
                            frame, x, y, color=(255, 0, 0), radius=5
                        )

            frames[frame_idx] = frame

        return frames