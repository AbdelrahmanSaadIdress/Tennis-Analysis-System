from ultralytics import YOLO
import numpy as np

class BallDetections:
    def __init__(self, model_id="Models/weights/best.pt"):
        self.model = YOLO(model_id)

    def detect_ball(self, frames, persons_dict, do_interpolation=True):
        detections = self.model.predict(frames)
        # ensure ball key exists
        persons_dict["ball"] = []

        for frame_num, frame_detections in enumerate(detections):

            frame_ball = {}  # empty if no ball detected

            for box in frame_detections.boxes:
                bbox = box.xyxy[0].tolist()

                frame_ball = {
                    "bbox": bbox
                }

                # for safety → break after the first ball.
                break
            persons_dict["ball"].append(frame_ball)

        # To be understandable.
        persons_and_ball_dict = persons_dict
        # Do the interpolation
        if do_interpolation:
            persons_and_ball_dict = self.interploate_ball_postions(persons_and_ball_dict)
        
        return persons_and_ball_dict

    def interploate_ball_postions(self, tracks):
        ball_tracks = tracks.get("ball", [])

        num_frames = len(ball_tracks)

        centers = np.full((num_frames, 2), np.nan)
        sizes = np.full((num_frames, 2), np.nan)  # (w, h)

        # 1️⃣ Extract centers + sizes
        for i, frame_ball in enumerate(ball_tracks):
            if "bbox" not in frame_ball:
                continue

            x1, y1, x2, y2 = frame_ball["bbox"]
            centers[i] = [(x1 + x2) / 2, (y1 + y2) / 2]
            sizes[i] = [x2 - x1, y2 - y1]

        # 2️⃣ Find valid indices
        valid = ~np.isnan(centers[:, 0])

        if valid.sum() < 2:
            return tracks  # not enough data to interpolate

        idx = np.arange(num_frames)

        # 3️⃣ Interpolate centers
        centers[:, 0] = np.interp(idx, idx[valid], centers[valid, 0])
        centers[:, 1] = np.interp(idx, idx[valid], centers[valid, 1])

        # 4️⃣ Interpolate sizes (optional but recommended)
        sizes[:, 0] = np.interp(idx, idx[valid], sizes[valid, 0])
        sizes[:, 1] = np.interp(idx, idx[valid], sizes[valid, 1])

        # 5️⃣ Rebuild bboxes
        for i in range(num_frames):
            cx, cy = centers[i]
            w, h = sizes[i]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            tracks["ball"][i] = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            }

        return tracks


