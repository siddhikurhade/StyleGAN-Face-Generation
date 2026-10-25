
import torch
import torch.nn as nn
import torch.nn.functional as F


# Maps random latent vectors (z) into intermediate latent vectors (w)
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=128, w_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim),
        )

    def forward(self, z):
        return self.net(z)


# Convolution layer with style modulation and optional random noise
class StyledConv(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim=128):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.norm = nn.InstanceNorm2d(out_channels)
        self.style = nn.Linear(w_dim, out_channels)

        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(
            torch.zeros(1, out_channels, 1, 1)
        )

    def forward(self, x, w, noise=True):
        x = self.conv(x)

        # Add fresh random noise when enabled
        if noise:
            random_noise = torch.randn(
                x.size(0),
                1,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )
            x = x + self.noise_strength * random_noise

        x = self.norm(x)

        # Apply style information
        scale = self.style(w).unsqueeze(-1).unsqueeze(-1)
        x = x * (1 + scale) + self.bias

        return F.leaky_relu(x, 0.2)


# StyleGAN-inspired generator
class Generator(nn.Module):
    def __init__(self, z_dim=128, w_dim=128):
        super().__init__()

        self.mapping = MappingNetwork(z_dim, w_dim)

        # Learnable constant input
        self.constant = nn.Parameter(
            torch.randn(1, 256, 4, 4)
        )

        # Styled convolution layers
        self.conv4 = StyledConv(256, 256, w_dim)
        self.conv8 = StyledConv(256, 256, w_dim)
        self.conv16 = StyledConv(256, 128, w_dim)
        self.conv32 = StyledConv(128, 64, w_dim)
        self.conv64 = StyledConv(64, 32, w_dim)

        # Convert feature maps to RGB
        self.to_rgb = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, z, mixing_z=None, noise=True):
        # Convert latent vectors into style vectors
        w1 = self.mapping(z)

        if mixing_z is not None:
            w2 = self.mapping(mixing_z)
        else:
            w2 = w1

        # Expand the constant for the batch
        x = self.constant.expand(z.size(0), -1, -1, -1)

        # 4x4 resolution
        x = self.conv4(x, w1, noise=noise)

        # 8x8 resolution
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv8(x, w1, noise=noise)

        # 16x16 resolution
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv16(x, w2, noise=noise)

        # 32x32 resolution
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv32(x, w2, noise=noise)

        # 64x64 resolution
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv64(x, w2, noise=noise)

        # Output RGB image in the range [-1, 1]
        image = torch.tanh(self.to_rgb(x))

        return image


# Discriminator: distinguishes real images from generated images
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
        )

    def forward(self, image):
        return self.net(image)
