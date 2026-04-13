import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from models import AutoencoderKL, ModernDiffusionUNet
from datasets.text_image import TextImageJsonl
from utils.text_encoder import load_clip, encode_text


def make_beta_schedule(num_steps=1000, beta_start=1e-4, beta_end=0.02):
    # 线性 beta 调度，训练与采样保持一致
    return torch.linspace(beta_start, beta_end, num_steps)


class DiffusionSchedule:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        # 初始化加噪参数
        self.betas = make_beta_schedule(num_steps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 构建前一时刻的alphas_cumprod序列
        # 1.0 + [:-1]，去掉原序列最后一位
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]
        ], dim=0)

    def sample_timesteps(self, batch_size):
        # (batch_size,) 为 batch 中每个样本均匀随机抽取一个0-（len-1）的时间步 t，最终返回[B]
        # torch.randint()取[)，无需-1
        return torch.randint(0, len(self.betas), (batch_size,), device=self.betas.device)

# 无文本+图片训练的数据加载器
def build_dataloader(data_root, resolution, batch_size, num_workers):
    # 简单 ImageFolder 数据加载器用于无文本训练
    tfm = transforms.Compose([
        # BILINEAR为缩放图片时使用双线性插值
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    # .ImageFolder()遍历路径下的图片并记录路径，读取时经过trnasform处理
    # 但必须注意，图片并不直接放在data_root下而是放在其子文件夹中
    # ImageFolder扫描 data_root 下的一级子文件夹名作为类别名，返回tuple(image, class_idx)
    ds = datasets.ImageFolder(data_root, transform=tfm)
    # num_workers指定线程池数
    # pin_memory=True用于将数据锁定在内存中不换页至硬盘，用于提速
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# 文本+图片训练的数据加载器
def build_jsonl_dataloader(jsonl_path, img_root, resolution, batch_size, num_workers):
    # jsonl 数据集可同时提供图像与文本 caption，用于条件训练
    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    # 自定义Dataset类型实例
    ds = TextImageJsonl(jsonl_path, img_root=img_root, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def diffusion_loss(model, vae, schedule: DiffusionSchedule, x0, context=None):
    # 一轮训练的损失计算：UNet 预测噪声，MSE 为主损失，附加小权重的 posterior KL 正则
    device = x0.device
    # 端到端训练时不冻结 VAE，方便联合优化
    # 编码得 latent z 与后验分布
    z, posterior = vae.encode(x0)  
    # 取batch_size
    bsz = z.size(0)
    # t代表每个样本对应的时间步[B]
    t = schedule.sample_timesteps(bsz)
    noise = torch.randn_like(z)  # 真实噪声

    # x_t = sqrt(alpha_bar) * z + sqrt(1-alpha_bar) * noise
    # alphas_cumprod形状为[B]，需扩展至[B,1,1,1]
    alpha_bar = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
    # z加噪后
    noisy_z = torch.sqrt(alpha_bar) * z + torch.sqrt(1 - alpha_bar) * noise

    # UNet 预测噪声，pred依旧是[B,4,32,32]
    pred = model(noisy_z, t, context)  
    mse = F.mse_loss(pred, noise, reduction="mean")

    # posterior KL 作为辅助正则项（权重很小）
    kl = posterior.kl().mean()
    return mse + 1e-4 * kl, {"mse": mse.item(), "kl": kl.item()}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优先加载 jsonl（带 caption），否则使用 ImageFolder
    if args.jsonl is not None:
        dataloader = build_jsonl_dataloader(args.jsonl, args.img_root, args.resolution, args.batch_size, args.num_workers)
    else:
        dataloader = build_dataloader(args.data_root, args.resolution, args.batch_size, args.num_workers)

    vae = AutoencoderKL().to(device)
    unet = ModernDiffusionUNet().to(device)

    # 文本编码器（可选）用于将 caption 编码为 context
    text_encoder = None
    tokenizer = None
    if args.jsonl is not None:
        text_encoder, tokenizer = load_clip(device=device)

    # 实例化加噪方式
    schedule = DiffusionSchedule(args.num_steps, args.beta_start, args.beta_end, device)

    # 同时训练unet和vae的参数
    params = list(unet.parameters()) + list(vae.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # 混合精度优化，gradient scaling
    # 优化成float16能减少显存占用但会导致精度损失，需要scale避免损失，详见父类：
    # https://github.com/pytorch/pytorch/blob/v2.11.0/torch/amp/grad_scaler.py
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    #新建输出目录，逐轮训练
    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            # 若包含文字则batch为字典类型
            if args.jsonl is not None:
                # 获取图片和文字
                x = batch["pixel_values"].to(device)
                captions = batch["caption"]
                # classifier-free guidance 的训练：随机丢弃部分文本，使模型学会无条件生成
                if text_encoder is not None:
                    text_feat = encode_text(text_encoder, tokenizer, captions, device=device)
                    # drop_mask得到[B]的[0,1,0,1,...]布尔序列
                    # 对 batch 里每个样本独立采样一个随机数，若小于 caption_drop 就置为 True
                    #  True 表示这个样本要丢文本条件
                    drop_mask = torch.rand(len(captions), device=device) < args.caption_drop
                    # ~取反，将掩码作用于文本特征实现随机丢弃，并扩展成[B,1]
                    # encode_text()输出[B, hidden_dim]，此处 [B, hidden_dim] * [B, 1](broadcast)，逐元素相乘
                    text_feat = text_feat * ((~drop_mask).float().unsqueeze(1))
                else:
                    text_feat = None
                context = text_feat
            # 不含文本信息时为元组，（image, class_idx）
            else:
                x, _ = batch
                x = x.to(device)
                context = None

            # 优化过程
            # 清空梯度
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                # loss: mse + weight*kl
                # status: string
                loss, stats = diffusion_loss(unet, vae, schedule, x, context=context)
            # scale/step/update是父类torch.amp.GradSclaler类下的方法，.backward()为torch.Tensor类下的方法
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 训练进度打印
            if global_step % args.log_interval == 0:
                print(f"step {global_step} | loss {loss.item():.4f} | mse {stats['mse']:.4f} | kl {stats['kl']:.4f}")
            # 定步保存ckpt
            if global_step % args.ckpt_interval == 0 and global_step > 0:
                save_ckpt(args, vae, unet, optimizer, global_step)
            global_step += 1

            # 防止越界
            if global_step >= args.max_steps:
                break
        # 防止越界
        if global_step >= args.max_steps:
            break

    save_ckpt(args, vae, unet, optimizer, global_step, final=True)


def save_ckpt(args, vae, unet, optimizer, step, final=False):
    # 根据轮数决定文件名
    tag = "final" if final else f"step{step}"
    # 保存路径/文件名
    path = Path(args.out_dir) / f"ckpt_{tag}.pt"
    torch.save({
        "vae": vae.state_dict(),
        "unet": unet.state_dict(),
        "opt": optimizer.state_dict(),
        "step": step,
    }, path)
    print(f"[ckpt] saved to {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None, help="ImageFolder 根目录（无文本时用）")
    p.add_argument("--jsonl", type=str, default=None, help="jsonl 标注文件路径（有文本时用）")
    p.add_argument("--img_root", type=str, default=None, help="jsonl 中 image 字段的相对根目录")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--num_steps", type=int, default=1000, help="扩散时间步数 T")
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", help="开启 AMP 混合精度")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--caption_drop", type=float, default=0.1, help="classifier-free guidance 文本丢弃概率")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


