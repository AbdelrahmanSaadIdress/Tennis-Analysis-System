from collections import defaultdict
import math


from collections import defaultdict
import math


class PlayersAnalysis:
    def __init__(self, frame_rate=30, smooth_window=5):
        self.frame_rate = frame_rate
        self.smooth_window = smooth_window

    @staticmethod
    def measure_distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def add_speed_and_distance_to_tracks(self, tracks, transformed_tracks=None):
        players_tracks = tracks["players"]
        n_frames = len(players_tracks)

        total_distance = defaultdict(float)
        speed_history = defaultdict(list)

        for f in range(1, n_frames):
            prev_frame = players_tracks[f - 1]
            curr_frame = players_tracks[f]

            if not prev_frame or not curr_frame:
                continue

            for pid, pdata in curr_frame.items():
                if pid not in prev_frame:
                    continue

                p_prev = prev_frame[pid].get("player_position_in_meters")
                p_curr = pdata.get("player_position_in_meters")

                if p_prev is None or p_curr is None:
                    continue

                # ---- Physics-correct instantaneous speed ----
                dist_m = self.measure_distance(p_prev, p_curr)
                speed_kmh = dist_m * self.frame_rate * 3.6

                total_distance[pid] += dist_m
                speed_history[pid].append(speed_kmh)

                # ---- Smoothed speed ----
                recent = speed_history[pid][-self.smooth_window:]
                smooth_speed = sum(recent) / len(recent)
                avg_speed = sum(speed_history[pid]) / len(speed_history[pid])

                # ---- Write directly into tracks ----
                pdata["speed"] = smooth_speed
                pdata["raw_speed"] = speed_kmh
                pdata["avg_speed"] = avg_speed
                pdata["distance_m"] = total_distance[pid]

        return tracks
