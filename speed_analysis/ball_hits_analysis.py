import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


import pandas as pd
import numpy as np


def ball_tracks_to_dataframe_with_motion(tracks,new_tracks, court=None, smooth_window=5, eps=1e-4):
    ball_tracks = tracks.get("ball", [])
    data = []

    for frame_id, ball_data in enumerate(ball_tracks):
        if ball_data and "ball_position_in_meters" in ball_data:
            x, y = ball_data["ball_position_in_meters"]
            detected = True
        else:
            x, y = None, None
            detected = False

        data.append({
            "frame_id": frame_id,
            "x": x,
            "y": y,
            "detected": detected
        })

    df = pd.DataFrame(data)

    # ---- Interpolate missing detections ----
    df[["x", "y"]] = df[["x", "y"]].interpolate(limit_direction="both")

    # ---- Smooth ----
    df["x_smooth"] = df["x"].rolling(smooth_window, center=True, min_periods=1).mean()
    df["y_smooth"] = df["y"].rolling(smooth_window, center=True, min_periods=1).mean()

    # ---- Velocity (m/frame) ----
    df["dx"] = df["x_smooth"].diff()
    df["dy"] = df["y_smooth"].diff()
    df["speed_m_frame"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

    # ---- Direction ----
    df["dy_sign"] = np.sign(df["dy"])
    df.loc[df["dy"].abs() < eps, "dy_sign"] = 0

    # ---- Hit detection ----
    hit = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)-2):
        if df.loc[i-1, "dy_sign"] != 0 and df.loc[i+1, "dy_sign"] != 0:
            if df.loc[i-1, "dy_sign"] != df.loc[i+1, "dy_sign"]:
                hit[i] = True

    # ---- Collapse clusters ----
    hit_idx = np.where(hit)[0]
    clusters = np.split(hit_idx, np.where(np.diff(hit_idx) != 1)[0] + 1)
    clean_hit = np.zeros(len(df), dtype=bool)
    for c in clusters:
        if len(c):
            best = c[np.argmax(np.abs(df.loc[c, "dy"]))]
            clean_hit[best] = True

    df["hit"] = clean_hit

    # ---- Hitter (court side inferred) ----
    net_y = df["y_smooth"].median()
    hitter = []
    for i in range(len(df)):
        if not df.loc[i, "hit"]:
            hitter.append(None)
        else:
            hitter.append("player_1" if df.loc[i, "y_smooth"] > net_y else "player_2")
    df["hitter"] = hitter

    return df

def plot_ball_positions_and_diff_separately(df_ball, save_path_positions=None, save_path_diff=None):
    frames = df_ball["frame_id"]
    hit_frames = df_ball.loc[df_ball["hit"], "frame_id"]

    # Y position plot
    plt.figure(figsize=(14, 6))
    plt.plot(frames, df_ball["y"], "b.-", label="Raw Y")
    plt.plot(frames, df_ball["y_smooth"], "r.-", label="Smoothed Y")

    for f in hit_frames:
        plt.axvline(f, color="red", linestyle="--", alpha=0.6)
        y = df_ball.loc[df_ball["frame_id"] == f, "y_smooth"].values[0]
        player = df_ball.loc[df_ball["frame_id"] == f, "hitter"].values[0]
        plt.text(f, y, player, color="red", fontsize=9, rotation=90, va="bottom")

    plt.xlabel("Frame")
    plt.ylabel("Meters")
    plt.title("Ball Y Position with Detected Hits")
    plt.legend()
    plt.grid(True)

    if save_path_positions:
        plt.savefig(save_path_positions, dpi=300)
    plt.show()

    # dy plot
    plt.figure(figsize=(14, 6))
    plt.plot(frames, df_ball["dy"], "g.-", label="dy")
    for f in hit_frames:
        plt.axvline(f, color="red", linestyle="--", alpha=0.6)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("ΔMeters")
    plt.title("Vertical Velocity (dy) with Hit Frames")
    plt.legend()
    plt.grid(True)

    if save_path_diff:
        plt.savefig(save_path_diff, dpi=300)
    plt.show()

def save_outputs(df_ball, plot_base_name="ball_hits"):
    os.makedirs("Outputs/Output_DataFrames", exist_ok=True)
    os.makedirs("Outputs/Output_Graphs", exist_ok=True)

    df_path = os.path.join("Outputs/Output_DataFrames", f"{plot_base_name}.csv")
    df_ball.to_csv(df_path, index=False)
    print(f"✅ DataFrame saved to {df_path}")

    pos_plot_path = os.path.join("Outputs/Output_Graphs", f"{plot_base_name}_position.png")
    dy_plot_path  = os.path.join("Outputs/Output_Graphs", f"{plot_base_name}_dy.png")
    plot_ball_positions_and_diff_separately(df_ball, pos_plot_path, dy_plot_path)
    print(f"✅ Graphs saved to Outputs/Output_Graphs")

