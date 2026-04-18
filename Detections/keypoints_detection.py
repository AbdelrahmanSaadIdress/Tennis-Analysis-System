from Models.KeyPointsModel import KeypointsModel  
import torch
import cv2
import numpy as np

class KeypointsDetection:
    def __init__(self, 
                model_path="Models/weights/keypoints_best.pth",
                device="cuda",
                img_size=224,
                num_keypoints=14):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.num_keypoints = num_keypoints

        # Load model
        self.model = KeypointsModel(num_keypoints=self.num_keypoints).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✅ Model loaded successfully.")

    # ------------------- Preprocessing -------------------
    def preprocess_image(self, image = None, image_path=None):
        if image is None and image_path is None:
            raise ValueError("You must provide either 'image' or 'image_path'.")

        # Read from path if needed
        if image is None:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")

        # Convert to RGB if it’s BGR
        if image.shape[2] == 3 and np.max(image) > 1.0:  # assume BGR 0-255
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_img = image.copy()

        # Resize
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # Convert to tensor (C,H,W)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)

        return img_tensor, original_img


    # ------------------- Predict keypoints -------------------
    def predict_keypoints(self, image_tensor):
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.cpu().numpy().reshape(-1, 2)
        keypoints[:, 0] *= self.img_size
        keypoints[:, 1] *= self.img_size
        
        self.keypoints = keypoints
        return keypoints

    # ------------------- Draw keypoints as overlay -------------------
    def draw_keypoints(self, keypoints, image_shape, scale_x=1.0, scale_y=1.0):
        """
        Create a transparent overlay with keypoints.

        Args:
            keypoints: np.ndarray or list of (x, y)
            image_shape: tuple (H, W, C) of the target frame
            scale_x: float, scale factor for x coordinates
            scale_y: float, scale factor for y coordinates

        Returns:
            overlay: np.ndarray (H, W, 3), black background with keypoints drawn in color
        """
        overlay = np.zeros(image_shape, dtype=np.uint8)

        for (x, y) in keypoints:
            px = int(x * scale_x)
            py = int(y * scale_y)
            cv2.circle(overlay, (px, py), 10, (255, 0, 0), -1)  # red keypoints

        return overlay
    
        # How to use 
        """# Suppose you have a frame
            frame = cv2.imread("frame.png")

            # Get overlay
            overlay = detector.draw_keypoints(detector.keypoints, frame.shape)

            # Blend overlay on frame
            alpha = 0.6
            combined = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)

            cv2.imshow("Frame with keypoints overlay", combined)
            cv2.waitKey(0)
        """
        

    def add_to_annotations_dict(self, persons_and_ball_dict):
        """
        Adds the predicted court keypoints to the persons_and_ball_dict.

        Args:
            persons_and_ball_dict: dict
                {
                    "persons": [...],
                    "ball": [...]
                }

        Returns:
            persons_and_ball_dict: dict
                {
                    "persons": [...],          # unchanged
                    "ball": [...],             # unchanged
                    "court_keypoints": [[x1,y1], [x2,y2], ...]  # from self.keypoints
                }
        """

        # Ensure court_keypoints is a list of lists
        persons_and_ball_dict["court_keypoints"] = self.keypoints.tolist()

        return persons_and_ball_dict


    def _scale_keypoints(self, frame, model_size=224):
        
        h, w = frame.shape[:2]
        sx = w / model_size
        sy = h / model_size
        return np.array(
            [[x * sx, y * sy] for x, y in self.keypoints],
            dtype=np.float32
        )
