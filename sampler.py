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


def make_beta_schedule(num_schedule_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    # 与训练一致的线性 beta 调度，用于重建 α 表。
    return torch.linspace(beta_start, beta_end, num_schedule_timesteps)


class DDIMSampler:
    # DDIM 采样器：在 latent 空间迭代去噪，再用 VAE 解码。

    # DDIM默认跳步加速
    def __init__(self, model, num_sampling_steps=50, eta=0.0, beta_start=1e-4, beta_end=0.02, num_schedule_timesteps=1000, device="cpu"):
        self.model = model  
        self.num_sampling_steps = num_sampling_steps  
        # 随机性系数（0 表示确定性）
        # sample()迭代预测x_prev时是否引入额外噪声的控制器
        self.eta = eta  
        self.device = device

        # 构造基础表，再子采样到 num_sampling_steps
        betas = make_beta_schedule(num_schedule_timesteps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.ddim_timesteps = torch.linspace(0, num_schedule_timesteps - 1, num_sampling_steps, dtype=torch.long, device=device)
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
        # 是否有图生图的输入
        start_latent=None,
    ):
        
        # latent_shape: 256 分辨率对应 (4,32,32)。
        # start_latent: 提供则从该 latent 开始（图生图），否则从随机噪声开始。
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
            if i == self.num_sampling_steps - 1:  # 最后一步
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
    # weights_only=True用于提防恶意代码
    # torch.load()返回的是字典
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # 实例化两个空的模型对象
    vae = AutoencoderKL().to(device)
    unet = ModernDiffusionUNet().to(device)
    # 把参数从字典中提取出来加载到空模型上
    vae.load_state_dict(ckpt["vae"])
    unet.load_state_dict(ckpt["unet"])
    # 只用不练
    vae.eval()
    unet.eval()
    return vae, unet


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", type=str, default=None, required=False, help=None, action = 'store_true')
    # 用于在命令行指定python sampler.py --lr 0.3 --resume
    # action是一个比较独特的参数，比如指定parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    # 执行时指定python sampler.py --lr 0.3 --resume，才有args.resume = True，否则为args.resume = Flase
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
    # 实例化对象为args，后续可通过args.ckpt等来访问
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, unet = load_models(args.ckpt, device)

    # 文本条件（可选）文生图需要prompt
    context = None
    # 若命令行有提供文本输入则覆盖None
    if args.prompt is not None:
        # 引入CLIP和context_encoder
        load_clip, encode_text = _lazy_clip()
        if load_clip is None:
            raise ImportError("需要 open_clip_torch 支持 prompt, 请先 pip install open_clip_torch")
        text_encoder, tokenizer = load_clip(device=device)
        context = encode_text(text_encoder, tokenizer, [args.prompt] * args.n, device=device)
    
    # 实例化采样器对象
    sampler = DDIMSampler(
        unet, 
        num_sampling_steps=args.steps, 
        eta=args.eta, 
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_schedule_timesteps=args.num_schedule_timesteps,
        device=device
    )

    # 图生图起始 latent（简化：直接加噪若 strength>0）
    init_latent = None
    latent_shape = (4, args.latent_h, args.latent_w)
    # 若命令行有提供图生图的输入则覆盖None
    if args.init_image:
        from PIL import Image
        from torchvision import transforms

        tfm = transforms.Compose(
            [   # HW短边缩放到256，长边按比例缩放
                transforms.Resize(args.latent_h * 8),
                # resize()后的图像中心出发裁切 latent_h * 8 的正方形
                transforms.CenterCrop(args.latent_h * 8),
                # [h,w,c]转变为[c,h,w]并且每通道/255缩放到 [0, 1] ，
                transforms.ToTensor(),
            ]
        )
        # 输入图片本质是提供图片路径的字符串，讲RGBA舍弃A转化为RGB，unsqueeze(0)转化为[1,c,h,w]
        img = tfm(Image.open(args.init_image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            # [1,3,256,256] -> [1,4,32,32]，转化到latent space
            z, _ = vae.encode(img)
        if args.strength > 0:
            # 根据 strength 叠加噪声作为近似起点
            # 由strength决定我们以多大的强度给原图新增噪声，strength为1.0表示原图覆盖为纯噪声，相当于退化为文生图，图片输入无效
            noise_steps = int(args.steps * args.strength)
            # 若strength不为1.0我们则无需在加噪、采样时跑满steps，仅需加噪至、去噪从-noise_steps开始
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
    # vae的输出最后被 tanh 压缩到[-1,1]，因此需映射回 [0,1]
    imgs = (imgs + 1) * 0.5

    # ------输出保存处理------
    # 把 args.out 统一当成“输出目标”来处理：
    # 如带扩展名，当成一个文件名前缀；如不带，直接当成目录名。
    out_path = Path(args.out)
    if out_path.suffix:
        # 例如 samples.png -> 输出目录 samples/，文件前缀 samples
        out_dir = out_path.with_suffix("")
        file_prefix = out_path.stem
    else:
        # 例如 samples -> 输出目录 samples/，文件前缀 samples
        out_dir = out_path
        file_prefix = out_path.name

    # 先创建输出目录，避免后面保存时路径不存在。
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐张保存 batch 中的图片，文件名使用三位序号避免覆盖。
    for idx, img in enumerate(imgs):
        save_path = out_dir / f"{file_prefix}_{idx:03d}.png"
        utils.save_image(img, save_path)

    # 输出最终保存位置和保存数量，便于命令行确认结果。
    print(f"saved {imgs.size(0)} images to {out_dir}")


if __name__ == "__main__":
    main()
