from torch import nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


def create_effnetb3_model(num_classes):
    """Recreate the deployed architecture before loading its complete state dict."""
    model = efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(1536, num_classes))
    return model, EfficientNet_B3_Weights.DEFAULT.transforms()
