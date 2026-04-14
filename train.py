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

from models import ModernDiffusionUNet
from utils.text_image import TextImageJsonl
from utils.text_encoder import load_clip, encode_text
from utils.vae_adapter import create_vae, extract_vae_state_dict


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
def build_dataloader(data_root, resolution, batch_size, num_workers, shuffle=True):
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
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# 文本+图片训练的数据加载器
def build_jsonl_dataloader(jsonl_path, img_root, resolution, batch_size, num_workers, shuffle=True):
     # jsonl 数据集可同时提供图像与文本 caption，用于条件训练
    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    # 自定义Dataset类型实例
    ds = TextImageJsonl(jsonl_path, img_root=img_root, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


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
    noise = torch.randn_like(z)  # 鐪熷疄鍣０

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


@torch.no_grad()
def evaluate(unet, vae, schedule, dataloader, device, max_batches=50, text_encoder=None, tokenizer=None, caption_drop=0.0):
    # 在验证集上计算平均 loss/mse/kl，不回传梯度。
    unet.eval()
    vae.eval()
    total_loss = total_mse = total_kl = 0.0
    n = 0
    for b_idx, batch in enumerate(dataloader):
        if b_idx >= max_batches:
            break
        if isinstance(batch, dict):  # jsonl
            x = batch["pixel_values"].to(device)
            captions = batch["caption"]
            if text_encoder is not None:
                text_feat = encode_text(text_encoder, tokenizer, captions, device=device)
                drop_mask = torch.rand(len(captions), device=device) < caption_drop
                text_feat = text_feat * ((~drop_mask).float().unsqueeze(1))
            else:
                text_feat = None
            context = text_feat
        else:
            x, _ = batch
            x = x.to(device)
            context = None
        loss, stats = diffusion_loss(unet, vae, schedule, x, context=context)
        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_mse += stats["mse"] * bsz
        total_kl += stats["kl"] * bsz
        n += bsz
    unet.train()
    vae.train()
    if n == 0:
        return {"loss": None, "mse": None, "kl": None}
    return {"loss": total_loss / n, "mse": total_mse / n, "kl": total_kl / n}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优先加载 jsonl（带 caption），否则使用 ImageFolder
    if args.jsonl is not None:
        dataloader = build_jsonl_dataloader(args.jsonl, args.img_root, args.resolution, args.batch_size, args.num_workers, shuffle=True)
    else:
        dataloader = build_dataloader(args.data_root, args.resolution, args.batch_size, args.num_workers, shuffle=True)

    # 验证集（可选）
    val_loader = None
    if args.val_jsonl is not None:
        val_loader = build_jsonl_dataloader(args.val_jsonl, args.val_img_root, args.resolution, args.batch_size, args.num_workers, shuffle=False)
    elif args.val_root is not None:
        val_loader = build_dataloader(args.val_root, args.resolution, args.batch_size, args.num_workers, shuffle=False)

    vae = create_vae(
        backend=args.vae_backend,
        device=device,
        source=args.vae_source if args.vae_backend == "diffusers" else None,
    )
    if args.vae_backend == "diffusers":
        print(f"[vae] backend=diffusers source={args.vae_source}")
    else:
        print("[vae] backend=toy")
    if args.vae_ckpt:
        # 允许加载外部预训练 VAE，strict=False 以兼容部分键名不一致
        vae_ckpt_obj = torch.load(args.vae_ckpt, map_location=device)
        load_info = vae.load_state_dict(extract_vae_state_dict(vae_ckpt_obj), strict=False)
        print(f"[vae] loaded from {args.vae_ckpt}, missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}")
    if args.freeze_vae:
        for p in vae.parameters():
            p.requires_grad = False
        print("[vae] parameters frozen (只训练 UNet)")

    unet = ModernDiffusionUNet().to(device)

    # 文本编码器（可选）用于将 caption 编码为 context
    text_encoder = None
    tokenizer = None
    if args.jsonl is not None:
        text_encoder, tokenizer = load_clip(device=device)

    # 实例化加噪方式
    schedule = DiffusionSchedule(args.num_steps, args.beta_start, args.beta_end, device)

    # 同时训练unet和vae的参数；若冻结 VAE 则只优化 UNet
    if args.freeze_vae:
        params = list(unet.parameters())
    else:
        params = list(unet.parameters()) + list(vae.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # 混合精度优化，gradient scaling
    # 优化成float16能减少显存占用但会导致精度损失，需要scale避免损失，详见父类：
    # https://github.com/pytorch/pytorch/blob/v2.11.0/torch/amp/grad_scaler.py
    # AMP：仅在 CUDA 设备上启用，避免 CPU 环境报错
    scaler = torch.amp.GradScaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=args.amp and device.type == "cuda",
    )

    #新建输出目录，逐轮训练
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = Path(args.out_dir) / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,split,loss,mse,kl\n", encoding="utf-8")
    # 最佳验证结果单独记录，避免与常规间隔日志混在一起
    best_log_path = Path(args.out_dir) / "best_val_log.csv"
    if not best_log_path.exists():
        best_log_path.write_text("step,loss,mse,kl,ckpt\n", encoding="utf-8")

    global_step = 0
    # 初始化最大用于验证集测试比较
    best_val = float("inf")
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

            # 混合精度优化
            autocast_ctx = torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda")
            with autocast_ctx:
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
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{global_step},train,{loss.item():.6f},{stats['mse']:.6f},{stats['kl']:.6f}\n")
            # 定步保存ckpt
            if global_step % args.ckpt_interval == 0 and global_step > 0:
                save_ckpt(args, vae, unet, optimizer, global_step)
            # 若验证集存在则定步检查模型在验证集上的表现
            if val_loader is not None and global_step % args.val_interval == 0 and global_step > 0:
                # 仅返回"loss": total_loss / n,    "mse": total_mse / n,   "kl": total_kl / n
                val_stats = evaluate(unet, vae, schedule, val_loader, device, max_batches=args.val_max_batches,
                                     text_encoder=text_encoder, tokenizer=tokenizer, caption_drop=args.caption_drop if args.jsonl else 0.0)
                if val_stats["loss"] is not None:
                    #打印返回值
                    print(f"[val] step {global_step} | loss {val_stats['loss']:.4f} | mse {val_stats['mse']:.4f} | kl {val_stats['kl']:.4f}")
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{global_step},val,{val_stats['loss']:.6f},{val_stats['mse']:.6f},{val_stats['kl']:.6f}\n")
                    # 寻找最优值，若更小则保存该模型
                    if val_stats["loss"] < best_val:
                        best_val = val_stats["loss"]
                        best_ckpt_path = save_ckpt(args, vae, unet, optimizer, global_step, final=False, best=True)
                        # 记录“当前最佳验证结果”到独立表，便于后续筛选 best step 与 best ckpt
                        with best_log_path.open("a", encoding="utf-8") as f:
                            f.write(f"{global_step},{val_stats['loss']:.6f},{val_stats['mse']:.6f},{val_stats['kl']:.6f},{best_ckpt_path}\n")
                        # 最佳验证集表现更新于第几步
                        print(f"[ckpt] best val updated at step {global_step}")
            global_step += 1

            # 防止越界
            if global_step >= args.max_steps:
                break
        # 防止越界
        if global_step >= args.max_steps:
            break

    save_ckpt(args, vae, unet, optimizer, global_step, final=True)


def save_ckpt(args, vae, unet, optimizer, step, final=False, best=False):
    # 根据轮数决定文件名
    tag = "final" if final else (f"best_step{step}" if best else f"step{step}")
    # 保存路径/文件名
    path = Path(args.out_dir) / f"ckpt_{tag}.pt"
    torch.save({
        "vae": vae.state_dict(),
        "unet": unet.state_dict(),
        "opt": optimizer.state_dict(),
        "step": step,
        "meta": {
            "vae_backend": args.vae_backend,
            "vae_source": args.vae_source,
        },
    }, path)
    print(f"[ckpt] saved to {path}")
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None, help="无条件训练的 ImageFolder 根目录")
    p.add_argument("--jsonl", type=str, default=None, help="文图训练使用的 jsonl 标注文件路径")
    p.add_argument("--img_root", type=str, default=None, help="jsonl 中图片相对路径对应的根目录")
    p.add_argument("--val_root", type=str, default=None, help="验证集 ImageFolder 根目录")
    p.add_argument("--val_jsonl", type=str, default=None, help="验证集 jsonl 标注文件路径")
    p.add_argument("--val_img_root", type=str, default=None, help="验证集 jsonl 图片根目录")
    p.add_argument("--val_interval", type=int, default=1000, help="每 N 个 step 进行一次验证")
    p.add_argument("--val_max_batches", type=int, default=50, help="每次验证最多评估的 batch 数")
    p.add_argument("--resolution", type=int, default=256, help="训练输入分辨率")
    p.add_argument("--batch_size", type=int, default=4, help="训练 batch 大小")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader 进程数")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数上限")
    p.add_argument("--max_steps", type=int, default=10000, help="训练总 step 上限")
    p.add_argument("--num_steps", type=int, default=1000, help="扩散过程总时间步 T")
    p.add_argument("--beta_start", type=float, default=1e-4, help="beta 调度起始值")
    p.add_argument("--beta_end", type=float, default=0.02, help="beta 调度结束值")
    p.add_argument("--lr", type=float, default=1e-4, help="优化器学习率")
    p.add_argument("--amp", action="store_true", help="启用 AMP 混合精度")
    p.add_argument("--out_dir", type=str, default="runs", help="训练输出目录（日志与 ckpt）")
    p.add_argument("--log_interval", type=int, default=50, help="每 N 个 step 打印一次训练日志")
    p.add_argument("--ckpt_interval", type=int, default=1000, help="每 N 个 step 保存一次常规 ckpt")
    p.add_argument("--caption_drop", type=float, default=0.1, help="CFG 训练时文本条件随机丢弃概率")
    p.add_argument("--vae_backend", type=str, default="toy", choices=["toy", "diffusers"], help="VAE 后端：toy 或 diffusers")
    p.add_argument("--vae_source", type=str, default="stabilityai/sd-vae-ft-mse", help="diffusers VAE 来源（HF 模型名或本地目录）")
    p.add_argument("--vae_ckpt", type=str, default=None, help="训练开始前加载的 VAE 权重文件（state_dict）")
    p.add_argument("--freeze_vae", action="store_true", help="冻结 VAE，仅训练 UNet")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


