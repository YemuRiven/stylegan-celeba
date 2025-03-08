import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Mapping Network: 将输入噪声 z 转换为中间向量 w
# ----------------------------------------
class MappingNetwork(nn.Module):
    def __init__(self, nz=100, w_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(nz, w_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(w_dim, w_dim))
            layers.append(nn.ReLU(inplace=True))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        w = self.mapping(z)
        return w

# ----------------------------------------
# AdaIN 模块: 对特征图进行调制 (简化版本)
# ----------------------------------------
class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super(AdaIN, self).__init__()
        self.scale = nn.Linear(w_dim, channels)
        self.bias = nn.Linear(w_dim, channels)

    def forward(self, x, w):
        # x: (B, C, H, W)
        # w: (B, w_dim)
        scale = self.scale(w).unsqueeze(2).unsqueeze(3)
        bias = self.bias(w).unsqueeze(2).unsqueeze(3)
        # 对 x 做 Instance Norm
        x_norm = F.instance_norm(x)
        return scale * x_norm + bias

# ----------------------------------------
# Synthesis Block: 一层上采样与调制卷积块
# ----------------------------------------
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(SynthesisBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, w)
        x = self.activation(x)
        return x

# ----------------------------------------
# Synthesis Network: 从固定输入生成图像
# ----------------------------------------
class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim=512, image_channels=3, feature_map_base=512):
        super(SynthesisNetwork, self).__init__()
        # 固定初始输入, 从 4x4 开始
        self.const_input = nn.Parameter(torch.randn(1, feature_map_base, 4, 4))
        # 逐步上采样块 (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64)
        self.block1 = SynthesisBlock(feature_map_base, feature_map_base//2, w_dim)
        self.block2 = SynthesisBlock(feature_map_base//2, feature_map_base//4, w_dim)
        self.block3 = SynthesisBlock(feature_map_base//4, feature_map_base//8, w_dim)
        self.block4 = SynthesisBlock(feature_map_base//8, feature_map_base//16, w_dim)
        # 输出层: 将特征映射到图像空间
        self.conv_out = nn.Conv2d(feature_map_base//16, image_channels, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, w):
        # w: (B, w_dim)
        batch_size = w.size(0)
        x = self.const_input.repeat(batch_size, 1, 1, 1)  # (B, 512, 4, 4)
        x = self.block1(x, w)  # (B, 256, 8, 8)
        x = self.block2(x, w)  # (B, 128, 16, 16)
        x = self.block3(x, w)  # (B, 64, 32, 32)
        x = self.block4(x, w)  # (B, 32, 64, 64)
        x = self.conv_out(x)   # (B, image_channels, 64, 64)
        x = self.tanh(x)       # 输出范围 [-1, 1]
        return x

# ----------------------------------------
# Generator: 包含 Mapping Network 与 Synthesis Network
# ----------------------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, w_dim=512, num_mapping_layers=8, image_channels=3, feature_map_base=512):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(nz, w_dim, num_mapping_layers)
        self.synthesis = SynthesisNetwork(w_dim, image_channels, feature_map_base)

    def forward(self, z):
        # z: (B, nz)
        w = self.mapping(z)
        img = self.synthesis(w)
        return img

# ----------------------------------------
# Discriminator: 简单的卷积判别器 (类似 DCGAN)
# ----------------------------------------
class Discriminator(nn.Module):
    def __init__(self, image_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: (B, image_channels, 64, 64)
            nn.Conv2d(image_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)          # shape: (B, 1, 5, 5)
        out = out.view(x.size(0), -1).mean(1)  # shape: (B,)
        return out