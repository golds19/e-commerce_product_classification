import torch
import torchvision

def setup_model(num_classes, device):
    # Load pre-trained weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Freeze the base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=num_classes,
                        bias=True).to(device)
    )

    return model