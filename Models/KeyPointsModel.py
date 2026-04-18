import torch
import torch.nn as nn
import torchvision.models as models


class KeypointsModel(nn.Module):
    def __init__(self, num_keypoints=14):
        super(KeypointsModel, self).__init__()

        # Load a pretrained ResNet backbone (e.g., resnet18)
        self.backbone = models.resnet18(pretrained=True)

        # Remove the final classification layer
        # ResNet has: fc = Linear(512, 1000)
        self.backbone.fc = nn.Identity()  # output: feature vector of size 512

        # Fully connected layer to predict all keypoints
        self.fc = nn.Linear(512, num_keypoints * 2)

    def forward(self, images):
        """
        images: Tensor of shape (B, 3, H, W)
        returns: Tensor of shape (B, num_keypoints*2)
        """
        features = self.backbone(images)  # shape: (B, 512)
        out = self.fc(features)           # shape: (B, num_keypoints*2)
        return out
