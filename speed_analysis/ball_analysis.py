import numpy as np
from .ball_hits_analysis import ball_tracks_to_dataframe_with_motion, save_outputs

class BallAnalysis:
    def __init__(self, tracks, new_tracks, court=None, fps=30):
        self.fps = fps
        self.df = ball_tracks_to_dataframe_with_motion(tracks, new_tracks, court)
        save_outputs(self.df)

    def speed(self):
        """
        Adds:
        - shot_speed_kmh
        - avg_shot_speed_kmh
        Only on hit frames.
        """
        df = self.df.copy()

        df["shot_speed_kmh"] = np.nan
        df["avg_shot_speed_kmh"] = np.nan

        speed_history = []

        hit_frames = df.index[df["hit"]].tolist()

        for i in range(len(hit_frames) - 1):
            f0 = hit_frames[i]
            f1 = hit_frames[i + 1]

            dx = df.loc[f0:f1, "x_smooth"].diff()
            dy = df.loc[f0:f1, "y_smooth"].diff()
            speed_m_s = np.sqrt(dx**2 + dy**2) * self.fps

            peak_speed = speed_m_s.max()
            speed_kmh = peak_speed * 3.6

            df.at[f0, "shot_speed_kmh"] = speed_kmh

            speed_history.append(speed_kmh)
            df.at[f0, "avg_shot_speed_kmh"] = np.mean(speed_history)

        self.df = df
        return df
