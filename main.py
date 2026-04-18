import os

from utils import read_video, save_video, save_tracks, load_tracks
from Detections import PersonsDetections, BallDetections, KeypointsDetection
from drawing import FramesAnnotations
from speed_analysis import BallAnalysis, PlayersAnalysis
from Court import PositionExtractor
from miniCourt import *

def main():
    # =========================
    # Load video
    # =========================
    frames = read_video()

    # =========================
    # Initialize detectors
    # =========================
    persons_detections = PersonsDetections()
    ball_detections = BallDetections()
    keypoints_detection = KeypointsDetection()

    # =========================
    # Load or compute tracks
    # =========================
    dict_path = "Outputs/Output_Dicts/tracks.pkl"

    if dict_path and os.path.exists(dict_path):
        tracks = load_tracks(dict_path)
    else:
        person_tracks = persons_detections.track_all_persons(frames[:100])
        tracks = ball_detections.detect_ball(frames[:100], person_tracks)

        # Optional: save tracks
        save_tracks(tracks, dict_path)

    # =========================
    # Keypoints detection
    # =========================
    img_tensor, original_img = keypoints_detection.preprocess_image(frames[0])
    keypoints = keypoints_detection.predict_keypoints(img_tensor)

    # Attach keypoints to tracks
    tracks = keypoints_detection.add_to_annotations_dict(tracks)

    # Filter static IDs
    tracks = persons_detections.filter_players_static_ids(frames, tracks)

    # =========================
    # Court extraction
    # =========================
    frames_annotations = FramesAnnotations(tracks, frames[:1])
    court = frames_annotations.court

    position_extractor = PositionExtractor(court)
    position_extractor.process_tracks(tracks)

    # Optional scaling
    keypoints = keypoints_detection._scale_keypoints(frames[0])

    # =========================
    # Mini court setup
    # =========================
    drawer = GrayRectangleDrawer(
        rect_width=390,
        rect_height=740,
        right_margin=30,
        top_margin=80
    )

    mini_frames = drawer.draw(frames[:100])

    transformer = CourtTrackTransformer(keypoints, drawer)
    transformed_tracks = transformer.transform_tracks(
        mini_frames,
        tracks
    )

    # =========================
    # Tactical visualization
    # =========================
    annotator = TacticalViewAnnotator(drawer)

    annotated_frames = []
    for idx, frame in enumerate(mini_frames):
        frame_result = transformed_tracks[idx]
        annotated_frame = annotator.draw_frame_annotations(frame.copy(), frame_result)
        annotated_frames.append(annotated_frame)

    save_video(
        annotated_frames,
        "Outputs/Output_Videos/result_with_minicourt.mp4"
    )

    # =========================
    # Update tracks with transformed data
    # =========================
    position_extractor = PositionExtractor(court)
    tracks = position_extractor.update_tracks(tracks, transformed_tracks)

    # =========================
    # Speed & ball analysis
    # =========================
    player_analysis = PlayersAnalysis()
    ball_analysis = BallAnalysis(tracks, transformed_tracks)

    df = ball_analysis.speed()

    tracks = player_analysis.add_speed_and_distance_to_tracks(
        tracks,
        transformed_tracks=transformed_tracks
    )

    # =========================
    # Final visualization
    # =========================
    vis_frames = frames_annotations.draw(annotated_frames, df)

    save_video(
        vis_frames,
        "Outputs/Output_Videos/result_with_analysis.mp4"
    )


if __name__ == "__main__":
    main()