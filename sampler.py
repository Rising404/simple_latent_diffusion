import argparse  # 解析命令行参数
import math
from pathlib import Path

import torch
from torchvision import utils

from models import AutoencoderKL, ModernDiffusionUNet


def _lazy_clip():
    # 按需导入 open_clip，未安装时允许纯无文本推理。
    try:
        from utils.text_encoder import load_clip, encode_text
        return load_clip, encode_text
    except ModuleNotFoundError:
        return None, None


def make_beta_schedule(init_schedule_steps=1000, beta_start=1e-4, beta_end=0.02):
    # 与训练一致的线性 beta 调度，用于重建 α 表。
    return torch.linspace(beta_start, beta_end, init_schedule_steps)


class DDIMSampler:
    # DDIM 采样器：在 latent 空间迭代去噪，再用 VAE 解码。

    # DDIM默认跳步加速，此处num_steps取50
    def __init__(self, model, num_steps=50, eta=0.0, device="cpu"):
        self.model = model  
        self.num_steps = num_steps  
        # 随机性系数（0 表示确定性）
        # sample()迭代预测x_prev时是否引入额外噪声的控制器
        self.eta = eta  
        self.device = device

        # 构造 1000 步基础表，再子采样到 num_steps
        betas = make_beta_schedule(1000).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.ddim_timesteps = torch.linspace(0, 999, num_steps, dtype=torch.long, device=device)
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_one_minus = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_alpha = torch.sqrt(alphas_cumprod)

    @torch.no_grad()
    def sample(
        self,
        vae,
        batch_size=1,
        latent_shape=(4, 32, 32),
        context=None,
        guidance_scale=1.0,
        start_latent=None,
    ):
        """
        latent_shape：256 分辨率对应 (4,32,32)。
        start_latent：提供则从该 latent 开始（图生图），否则从随机噪声开始。
        """
        x = start_latent if start_latent is not None else torch.randn(batch_size, *latent_shape, device=self.device)

        # 逆序时间步（t 从大到小）
        for i, t in enumerate(reversed(self.ddim_timesteps)):
            # 每轮其实只传某一个时间步，但shape依旧要从[1]广播对齐成网络输入t_tensor[B]
            t_tensor = torch.tensor([t] * x.size(0), device=self.device)
            alpha = self.alphas_cumprod[t]
            sqrt_alpha = self.sqrt_alpha[t]
            sqrt_one_minus = self.sqrt_one_minus[t]

            # 预测噪声 eps
            eps = self.model(x, t_tensor, context)
            # 是否引入条件控制
            # uncond/base，为偏离的基线
            # cond/target，为目标条件
            # guidance_scale 和 context 共同限制eps是否考虑文本控制
            if guidance_scale != 1.0 and context is not None:
                eps_cond = eps
                eps_uncond = self.model(x, t_tensor, context=None)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)  # CFG

            # 当前条件反推 x0
            pred_x0 = (x - sqrt_one_minus * eps) / sqrt_alpha

            # 准备迭代，sample本质是从头/某一时间步反推上一时间步
            # 计算上一时刻 x_{t-1}
            if i == self.num_steps - 1:  # 最后一步
                x_prev = pred_x0 * sqrt_alpha + sqrt_one_minus * eps
            else:
                # 提取下一时间步
                t_prev = self.ddim_timesteps[len(self.ddim_timesteps) - 1 -(i+1)]
                alpha_prev = self.alphas_cumprod[t_prev]
                sigma = self.eta * math.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
                noise = torch.randn_like(x) if sigma > 0 else 0.0
                # 根据当前eps损失和pred_x0推测下一步的输入x_prev
                x_prev = (
                    torch.sqrt(alpha_prev) * pred_x0
                    + torch.sqrt(1 - alpha_prev - sigma**2) * eps
                    + sigma * noise
                )
            x = x_prev

        imgs = vae.decode(x) 
        return imgs


def load_models(ckpt_path, device):
    # 载入 ckpt，返回 VAE 与 UNet。
    ckpt = torch.load(ckpt_path, map_location=device)
    vae = AutoencoderKL().to(device)
    unet = ModernDiffusionUNet().to(device)
    vae.load_state_dict(ckpt["vae"])
    unet.load_state_dict(ckpt["unet"])
    vae.eval()
    unet.eval()
    return vae, unet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="训练保存的 ckpt 路径")
    parser.add_argument("--out", type=str, default="samples.png")
    parser.add_argument("--n", type=int, default=4, help="生成张数")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--latent_h", type=int, default=32)
    parser.add_argument("--latent_w", type=int, default=32)
    parser.add_argument("--prompt", type=str, default=None, help="文生图 prompt")
    parser.add_argument("--init_image", type=str, default=None, help="图生图输入图")
    parser.add_argument("--strength", type=float, default=0.5, help="图生图加噪强度 0-1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, unet = load_models(args.ckpt, device)

    # 文本条件（可选）
    context = None
    if args.prompt is not None:
        load_clip, encode_text = _lazy_clip()
        if load_clip is None:
            raise ImportError("需要 open_clip_torch 支持 prompt，请先 pip install open_clip_torch")
        text_encoder, tokenizer = load_clip(device=device)
        context = encode_text(text_encoder, tokenizer, [args.prompt] * args.n, device=device)

    sampler = DDIMSampler(unet, num_steps=args.steps, eta=args.eta, device=device)

    # 图生图起始 latent（简化：直接加噪若 strength>0）
    init_latent = None
    latent_shape = (4, args.latent_h, args.latent_w)
    if args.init_image:
        from PIL import Image
        from torchvision import transforms

        tfm = transforms.Compose(
            [
                transforms.Resize(args.latent_h * 8),
                transforms.CenterCrop(args.latent_h * 8),
                transforms.ToTensor(),
            ]
        )
        img = tfm(Image.open(args.init_image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            z, _ = vae.encode(img)
        if args.strength > 0:
            # 根据 strength 叠加噪声作为近似起点
            noise_steps = int(args.steps * args.strength)
            for _ in range(noise_steps):
                z = z + torch.randn_like(z)
            sampler.ddim_timesteps = sampler.ddim_timesteps[-noise_steps:] if noise_steps > 0 else sampler.ddim_timesteps
        init_latent = z
        latent_shape = z.shape[1:]

    imgs = sampler.sample(
        vae,
        batch_size=args.n if init_latent is None else init_latent.size(0),
        latent_shape=latent_shape,
        context=context,
        guidance_scale=args.guidance,
        start_latent=init_latent,
    )
    imgs = (imgs + 1) * 0.5  # tanh 输出映射回 [0,1]
    utils.save_image(imgs, args.out, nrow=int(math.sqrt(args.n)))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
