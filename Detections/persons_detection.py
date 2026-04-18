import numpy as np
from ultralytics import YOLO

class PersonsDetections:
    def __init__(self, model_id="Models/weights/yolov8m.pt"):
        self.model = YOLO(model_id)

    def track_all_persons(self, frames):
        tracks = self.model.track(frames)
        persons_dict = {"persons": []}

        for frame_num, frame_detections in enumerate(tracks):
            frame_persons = {}  # persons in THIS frame

            if frame_detections.boxes.id is None:
                persons_dict["persons"].append(frame_persons)
                continue

            for box in frame_detections.boxes:
                # keep only persons
                if int(box.cls) != 0:
                    continue
                
                person_id = int(box.id)
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                frame_persons[person_id] = {
                    "bbox": bbox,
                    "person_id": person_id
                }

            persons_dict["persons"].append(frame_persons)

        return persons_dict

    # --------------------------------------------------
    # Filter players and store in a separate key "players"
    # --------------------------------------------------
    def filter_players_static_ids(self, frames, tracks):
        """
        Keep only two players per frame, closest to court keypoints.
        Assign static IDs: player_1 and player_2 across all frames.
        """
        persons_tracks = tracks.get("persons", [])
        keypoints = tracks.get("court_keypoints", [])

        if not keypoints:
            # fallback: return empty players
            tracks["players"] = [{} for _ in persons_tracks]
            return tracks

        # Scale keypoints to frame size
        MODEL_SIZE = 224
        H, W = frames[0].shape[:2]
        scale_x = W / MODEL_SIZE
        scale_y = H / MODEL_SIZE
        scaled_keypoints = [(x * scale_x, y * scale_y) for x, y in keypoints]
        keypoints_array = np.array(scaled_keypoints)

        players_tracks = []

        # For keeping the static IDs
        static_player_centers = [None, None]  # [player_1_center, player_2_center]

        for frame_idx, frame_persons in enumerate(persons_tracks):
            filtered_frame = {}

            if len(frame_persons) == 0:
                players_tracks.append(filtered_frame)
                continue

            # Compute min distance to court keypoints for each person
            person_distances = []
            person_centers = {}
            for pid, pdata in frame_persons.items():
                x1, y1, x2, y2 = pdata["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                person_centers[pid] = np.array([cx, cy])

                dists = np.linalg.norm(keypoints_array - np.array([cx, cy]), axis=1)
                min_dist = np.min(dists)
                person_distances.append((pid, min_dist))

            # Sort persons by min distance to keypoints
            person_distances.sort(key=lambda x: x[1])

            # Select the 2 closest persons
            if frame_idx == 0:
                # First frame → assign static_player_centers
                closest_pids = [pid for pid, _ in person_distances[:2]]
                for i, pid in enumerate(closest_pids):
                    filtered_frame[f"player_{i+1}"] = frame_persons[pid]
                    static_player_centers[i] = person_centers[pid]
            else:
                # Subsequent frames → assign based on nearest to previous center
                remaining = dict(frame_persons)  # copy
                assigned = [False, False]

                for pid, center in person_centers.items():
                    if len(remaining) == 0:
                        break

                    # Compute distance to static centers
                    d0 = np.linalg.norm(center - static_player_centers[0])
                    d1 = np.linalg.norm(center - static_player_centers[1])

                    if not assigned[0] and d0 < d1:
                        filtered_frame["player_1"] = frame_persons[pid]
                        static_player_centers[0] = center
                        assigned[0] = True
                    elif not assigned[1]:
                        filtered_frame["player_2"] = frame_persons[pid]
                        static_player_centers[1] = center
                        assigned[1] = True

                # Fallback: if only 1 player detected
                if "player_1" not in filtered_frame and person_distances:
                    pid = person_distances[0][0]
                    filtered_frame["player_1"] = frame_persons[pid]
                    static_player_centers[0] = person_centers[pid]
                if "player_2" not in filtered_frame and len(person_distances) > 1:
                    pid = person_distances[1][0]
                    filtered_frame["player_2"] = frame_persons[pid]
                    static_player_centers[1] = person_centers[pid]

            players_tracks.append(filtered_frame)

        # Save under tracks
        tracks["players"] = players_tracks
        return tracks



