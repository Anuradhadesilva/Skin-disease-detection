import torch.nn as nn
import torch

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
# scripts/unet_model.py
# import torch
# import torch.nn as nn

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, n_classes=1, in_channels=3):
#         super(UNet, self).__init__()

#         self.down1 = DoubleConv(in_channels, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.down2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(2)
#         self.down3 = DoubleConv(128, 256)
#         self.pool3 = nn.MaxPool2d(2)
#         self.down4 = DoubleConv(256, 512)
#         self.pool4 = nn.MaxPool2d(2)

#         self.middle = DoubleConv(512, 1024)

#         self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.conv4 = DoubleConv(1024, 512)
#         self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.conv3 = DoubleConv(512, 256)
#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.conv2 = DoubleConv(256, 128)
#         self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.conv1 = DoubleConv(128, 64)

#         self.output = nn.Conv2d(64, n_classes, kernel_size=1)

#     def forward(self, x):
#         c1 = self.down1(x)
#         p1 = self.pool1(c1)
#         c2 = self.down2(p1)
#         p2 = self.pool2(c2)
#         c3 = self.down3(p2)
#         p3 = self.pool3(c3)
#         c4 = self.down4(p3)
#         p4 = self.pool4(c4)

#         mid = self.middle(p4)

#         u4 = self.up4(mid)
#         u4 = torch.cat([u4, c4], dim=1)
#         c4 = self.conv4(u4)

#         u3 = self.up3(c4)
#         u3 = torch.cat([u3, c3], dim=1)
#         c3 = self.conv3(u3)

#         u2 = self.up2(c3)
#         u2 = torch.cat([u2, c2], dim=1)
#         c2 = self.conv2(u2)

#         u1 = self.up1(c2)
#         u1 = torch.cat([u1, c1], dim=1)
#         c1 = self.conv1(u1)

#         return torch.sigmoid(self.output(c1))
