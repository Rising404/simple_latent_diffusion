import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    # 管理高斯分布的均值与对数方差，便于采样与 KL 计算。

    def __init__(self, moments: torch.Tensor):
        # moments: [B, 2*C, H, W] -> 均值 与 对数方差
        # 2 * latent_channels，每个latent_ch都需要一个均值和一个方差
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        # torch.clamp() —— y_i ​= min( max( xi​, min ), max)
        # 小于-30取-30, 中间该是啥是啥，大于20取20
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)  # 防止数值发散

    # 引入重参数化技巧
    # 要从网络算出的这个正态分布 N(μ,σ^2) 中抽取一个随机的潜变量 Z 给 Decoder 去画图
    # 但直接采样参数梯度会断
    # 但被证明：和上述等价的是，可从标准正态分布 N(0,1) 中采样一个纯噪声 ϵ，然后做一次线性变换 Z=μ+σ×ϵ。
    def sample(self) -> torch.Tensor:
        # self.logvar的形状是[B, C, H, W]， std相同，eps也相同
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        # std eps形状相同，逐元素相乘
        return self.mean + std * eps  
    

    # other为"DiagonalGaussianDistribution"类，用字符串包裹类名延迟解释
    # 求出各样本的KL散度和用于引入正则约束
    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        # 计算 KL(q || p)，默认与标准正态 p:N(0,1)。
        if other is None:
            return 0.5 * torch.sum(
                torch.exp(self.logvar) + self.mean ** 2 - 1.0 - self.logvar,
                dim=[1, 2, 3],  # 各个样本单独处理
            ) # 返回各个样本在所有latent上的KL和
        
        # 不仅仅是供VAE和标准正态KL正则的工具，也提供了和其它高斯分布比较的KL工具
        # 此处为 KL(N_1 || N_2) 的
        else:
            return 0.5 * torch.sum(
                other.logvar - self.logvar
                + torch.exp(self.logvar - other.logvar)
                + (self.mean - other.mean) ** 2 / torch.exp(other.logvar)
                - 1.0,
                dim=[1, 2, 3],
            )


class ResBlock(nn.Module):
    # 相较于unet/ResnetBlock，输入不包括时间步（毕竟不在网络内），skip等效于short_cut
    # 残差块只改变通道大小，不改变图像分辨率
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 如果输入图像的通道不对齐则映射对齐，而后残差连接
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # 通道对齐

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)  # 残差


class Encoder(nn.Module):
    # encoder 返回一个对角高斯分布类，latent_ch mean + logvar
    # conv1(in_ch_to_base_ch) -> （残差+残差+像素/2卷积）* 3 -> conv2(base_ch*4_to_latent_ch*2) -> 对角高斯分布
    # 3 -> base_ch -> 2* -> 4* 【通道】，三次 2× 下采样得到 1/8 【分辨率】。
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
    # 反向对称：最近邻上采样 + 3×3 Conv，再接 ResBlock。
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
        # 输出前处理
        h = self.act_out(self.norm_out(h))
        return torch.tanh(self.conv_out(h))  # 限制到 [-1,1]，所以使用vae的输出时还需要额外缩放


class AutoencoderKL(nn.Module):
    # VAE 整体封装，除了完成vae自身压缩任务，也提供单独工具，查看某步结果
    # __init__用于指定vae的输入、中间特征维度、目标latent_ch、特定缩放因子

    def __init__(self, in_ch=3, base_ch=128, latent_ch=4, scaling_factor=0.18215):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, latent_ch)
        self.decoder = Decoder(in_ch, base_ch, latent_ch)
        self.scaling_factor = scaling_factor

    # （单独工具）: 返回后验分布 + 缩放 latent
    # 输出提供给diffusion pipeline
    def encode(self, x):
        posterior = self.encoder(x)
        z = posterior.sample() * self.scaling_factor  # 缩放 latent，严格来说需要根据数据集调整
        return z, posterior
   
    # （单独工具）: latent -> 像素
    # diffusion pipeline的输出喂给decoder还原
    def decode(self, z):
        z = z / self.scaling_factor  # 还原尺度
        return self.decoder(z)

    # 该forward主要服务于训练vae自身
    # 训练diffusion时常常冻结vae
    def forward(self, x):
        posterior = self.encoder(x)
        z = posterior.sample()
        rec = self.decoder(z)
        return rec, posterior
