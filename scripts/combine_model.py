# scripts/combine_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.enc1 = conv_block(3, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.conv_final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.conv_final(d1)

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetClassifier, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

class CombinedModel(nn.Module):
    def __init__(self, unet_model, classifier_model, freeze_unet=True):
        super(CombinedModel, self).__init__()
        self.unet = unet_model
        self.classifier = classifier_model
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False

    def forward(self, x):
        mask_logits = self.unet(x)  # (B,1,H,W)
        mask = torch.sigmoid(mask_logits)
        x_masked = x * mask  # mask image to focus classifier
        x_masked = F.interpolate(x_masked, size=(224,224), mode='bilinear', align_corners=False)
        logits = self.classifier(x_masked)
        return logits, mask_logits
   