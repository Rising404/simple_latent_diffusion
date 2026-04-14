import argparse
import math
from pathlib import Path

import torch
from torchvision import utils

from models import AutoencoderKL, ModernDiffusionUNet


def _lazy_clip():
    try:
        from utils.text_encoder import load_clip, encode_text
        return load_clip, encode_text
    except ModuleNotFoundError:
        return None, None


def make_beta_schedule(num_schedule_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_schedule_timesteps)


class DDIMSampler:
    def __init__(
        self,
        model,
        num_sampling_steps=50,
        eta=0.0,
        beta_start=1e-4,
        beta_end=0.02,
        num_schedule_timesteps=1000,
        device="cpu",
    ):
        self.model = model
        self.num_sampling_steps = num_sampling_steps
        self.eta = eta
        self.device = device

        betas = make_beta_schedule(num_schedule_timesteps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.ddim_timesteps = torch.linspace(
            0, num_schedule_timesteps - 1, num_sampling_steps, dtype=torch.long, device=device
        )
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
        x = start_latent if start_latent is not None else torch.randn(batch_size, *latent_shape, device=self.device)

        for i, t in enumerate(reversed(self.ddim_timesteps)):
            t_tensor = torch.tensor([t] * x.size(0), device=self.device)
            alpha = self.alphas_cumprod[t]
            sqrt_alpha = self.sqrt_alpha[t]
            sqrt_one_minus = self.sqrt_one_minus[t]

            eps = self.model(x, t_tensor, context)
            if guidance_scale != 1.0 and context is not None:
                eps_cond = eps
                eps_uncond = self.model(x, t_tensor, context=None)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            pred_x0 = (x - sqrt_one_minus * eps) / sqrt_alpha

            if i == self.num_sampling_steps - 1:
                x_prev = pred_x0 * sqrt_alpha + sqrt_one_minus * eps
            else:
                t_prev = self.ddim_timesteps[len(self.ddim_timesteps) - 1 - (i + 1)]
                alpha_prev = self.alphas_cumprod[t_prev]
                sigma = self.eta * math.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
                noise = torch.randn_like(x) if sigma > 0 else 0.0
                x_prev = (
                    torch.sqrt(alpha_prev) * pred_x0
                    + torch.sqrt(1 - alpha_prev - sigma ** 2) * eps
                    + sigma * noise
                )
            x = x_prev

        imgs = vae.decode(x)
        return imgs


def load_models(ckpt_path, device, vae_override=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vae = AutoencoderKL().to(device)
    unet = ModernDiffusionUNet().to(device)

    # Pre-alignment behavior: allow partial override loading.
    if vae_override:
        vae_ckpt = torch.load(vae_override, map_location=device)
        load_info = vae.load_state_dict(vae_ckpt, strict=False)
        print(
            f"[vae] override loaded from {vae_override}, "
            f"missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}"
        )
    else:
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
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta 调度起点")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta 调度终点")
    parser.add_argument("--num_schedule_timesteps", type=int, default=1000, help="Beta 调度总步数")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="外部 VAE state_dict 路径（推理阶段覆盖 ckpt 内置 VAE）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, unet = load_models(args.ckpt, device, vae_override=args.vae_ckpt)

    context = None
    if args.prompt is not None:
        load_clip, encode_text = _lazy_clip()
        if load_clip is None:
            raise ImportError("需要 open_clip_torch 支持 prompt，请先 pip install open_clip_torch")
        text_encoder, tokenizer = load_clip(device=device)
        context = encode_text(text_encoder, tokenizer, [args.prompt] * args.n, device=device)

    sampler = DDIMSampler(
        unet,
        num_sampling_steps=args.steps,
        eta=args.eta,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_schedule_timesteps=args.num_schedule_timesteps,
        device=device,
    )

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
    imgs = (imgs + 1) * 0.5

    out_path = Path(args.out)
    if out_path.suffix:
        out_dir = out_path.with_suffix("")
        file_prefix = out_path.stem
    else:
        out_dir = out_path
        file_prefix = out_path.name

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(imgs):
        save_path = out_dir / f"{file_prefix}_{idx:03d}.png"
        utils.save_image(img, save_path)

    print(f"saved {imgs.size(0)} images to {out_dir}")


if __name__ == "__main__":
    main()

