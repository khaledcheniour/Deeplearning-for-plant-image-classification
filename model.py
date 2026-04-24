import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention2D(nn.Module):
    def __init__(self, channels):
        super(SelfAttention2D, self).__init__()
        self.query = nn.Linear(channels, channels)
        self.key   = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.out   = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)

        attn = F.softmax(Q @ K.transpose(-2, -1) / (C ** 0.5), dim=-1)  # (B, N, N)
        out = attn @ V  # (B, N, C)

        out = self.out(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + out  # Residual connectio


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W)           # [B, C//8, H*W]
        key = self.key(x).view(B, -1, H * W)               # [B, C//8, H*W]
        value = self.value(x).view(B, C, H * W)            # [B, C, H*W]

        attn = torch.bmm(query.permute(0, 2, 1), key)      # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(value, attn.permute(0, 2, 1))      # [B, C, H*W]
        out = out.view(B, C, H, W)

        return self.gamma * out + x


class UNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = ConvBlock(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

        # Bridge
        self.bridge = ConvBlock(1024, 2048)
        self.attn = SelfAttention2D(2048)

        # Decoder
        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec1 = ConvBlock(2048, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec2 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ConvBlock(256, 128)
        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec5 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        s5 = self.enc5(self.pool4(s4))

        # Bridge + Self-Attention
        b = self.bridge(self.pool5(s5))
        b = self.attn(b)

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(b), s5], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s3], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), s2], dim=1))
        d5 = self.dec5(torch.cat([self.up5(d4), s1], dim=1))

        return torch.sigmoid(self.final(d5))


# Test run
model = UNetWithAttention()
output = model(torch.randn(1, 3, 256, 256))
print(output.shape)  # Should be: torch.Size([1, 1, 256, 256])
