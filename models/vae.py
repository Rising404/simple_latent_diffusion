import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    """管理高斯分布的均值与对数方差，便于采样与 KL 计算。"""

    def __init__(self, moments: torch.Tensor):
        # moments: [B, 2*C, H, W] -> 均值与 logvar
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)  # 防止数值发散

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mean + std * eps  # 重新参数化采样

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """计算 KL(q || p)，默认与标准正态。"""
        if other is None:
            return 0.5 * torch.sum(
                torch.exp(self.logvar) + self.mean ** 2 - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            return 0.5 * torch.sum(
                other.logvar - self.logvar
                + torch.exp(self.logvar - other.logvar)
                + (self.mean - other.mean) ** 2 / torch.exp(other.logvar)
                - 1.0,
                dim=[1, 2, 3],
            )


class ResBlock(nn.Module):
    """轻量 ResNet 块：GroupNorm + SiLU + Conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # 通道对齐

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)  # 残差


class Encoder(nn.Module):
    """3 -> base_ch -> 2* -> 4* 通道，三次 2× 下采样得到 1/8 分辨率。"""

    def __init__(self, in_ch=3, base_ch=128, latent_ch=4):
        super().__init__()
        ch = base_ch
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.block1 = nn.Sequential(  # H/2
            ResBlock(ch, ch),
            ResBlock(ch, ch),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(  # H/4
            ResBlock(ch, ch * 2),
            ResBlock(ch * 2, ch * 2),
            nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1),
        )
        self.block3 = nn.Sequential(  # H/8
            ResBlock(ch * 2, ch * 4),
            ResBlock(ch * 4, ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1),
        )
        self.conv_out = nn.Conv2d(ch * 4, latent_ch * 2, 3, padding=1)  # μ, logvar

    def forward(self, x):
        h = self.conv_in(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        moments = self.conv_out(h)
        return DiagonalGaussianDistribution(moments)


class Decoder(nn.Module):
    """反向对称：最近邻上采样 + 3×3 Conv，再接 ResBlock。"""

    def __init__(self, out_ch=3, base_ch=128, latent_ch=4):
        super().__init__()
        ch = base_ch * 4
        self.conv_in = nn.Conv2d(latent_ch, ch, 3, padding=1)
        self.block1 = nn.Sequential(  # 8x -> 16x
            ResBlock(ch, ch),
            ResBlock(ch, ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(  # 16x -> 32x
            ResBlock(ch, ch // 2),
            ResBlock(ch // 2, ch // 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch // 2, ch // 2, 3, padding=1),
        )
        self.block3 = nn.Sequential(  # 32x -> 64x
            ResBlock(ch // 2, ch // 4),
            ResBlock(ch // 4, ch // 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch // 4, ch // 4, 3, padding=1),
        )
        self.norm_out = nn.GroupNorm(32, ch // 4)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(ch // 4, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.act_out(self.norm_out(h))
        return torch.tanh(self.conv_out(h))  # 限制到 [-1,1]


class AutoencoderKL(nn.Module):
    """
    VAE 整体封装：
    - encode: 返回后验分布 + 缩放 latent
    - decode: latent -> 像素
    - scaling_factor: 与 Stable Diffusion 对齐的 0.18215
    """

    def __init__(self, in_ch=3, base_ch=128, latent_ch=4, scaling_factor=0.18215):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, latent_ch)
        self.decoder = Decoder(in_ch, base_ch, latent_ch)
        self.scaling_factor = scaling_factor

    def encode(self, x):
        posterior = self.encoder(x)
        z = posterior.sample() * self.scaling_factor  # 缩放 latent
        return z, posterior

    def decode(self, z):
        z = z / self.scaling_factor  # 还原尺度
        return self.decoder(z)

    def forward(self, x):
        posterior = self.encoder(x)
        z = posterior.sample()
        rec = self.decoder(z)
        return rec, posterior
