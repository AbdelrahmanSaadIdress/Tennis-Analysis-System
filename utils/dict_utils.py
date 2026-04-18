import pickle

# -----------------------------
# Save tracks to a file
# -----------------------------
def save_tracks(tracks, file_path="Outputs/Output_Dicts/tracks.pkl"):
    """
    Save the tracks dictionary to disk.
    
    Args:
        tracks: dict containing persons, players, ball, court_keypoints
        file_path: str, path to save the file (default 'tracks.pkl')
    """
    with open(file_path, "wb") as f:
        pickle.dump(tracks, f)
    print(f"✅ Tracks saved to '{file_path}'.")


# -----------------------------
# Load tracks from a file
# -----------------------------
def load_tracks(file_path="tracks.pkl"):
    """
    Load a tracks dictionary from disk.
    
    Args:
        file_path: str, path to the file
    
    Returns:
        tracks: dict
    """
    with open(file_path, "rb") as f:
        tracks = pickle.load(f)
    print(f"✅ Tracks loaded from '{file_path}'.")
    return tracks
