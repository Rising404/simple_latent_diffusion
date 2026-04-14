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
from utils.text_image import TextImageJsonl
from utils.text_encoder import load_clip, encode_text


def make_beta_schedule(num_steps=1000, beta_start=1e-4, beta_end=0.02):
    # 绾挎€?beta 璋冨害锛岃缁冧笌閲囨牱淇濇寔涓€鑷?
    return torch.linspace(beta_start, beta_end, num_steps)


class DiffusionSchedule:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        # 鍒濆鍖栧姞鍣弬鏁?
        self.betas = make_beta_schedule(num_steps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 鏋勫缓鍓嶄竴鏃跺埢鐨刟lphas_cumprod搴忓垪
        # 1.0 + [:-1]锛屽幓鎺夊師搴忓垪鏈€鍚庝竴浣?
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]
        ], dim=0)

    def sample_timesteps(self, batch_size):
        # (batch_size,) 涓?batch 涓瘡涓牱鏈潎鍖€闅忔満鎶藉彇涓€涓?-锛坙en-1锛夌殑鏃堕棿姝?t锛屾渶缁堣繑鍥瀃B]
        # torch.randint()鍙朳)锛屾棤闇€-1
        return torch.randint(0, len(self.betas), (batch_size,), device=self.betas.device)

# 鏃犳枃鏈?鍥剧墖璁粌鐨勬暟鎹姞杞藉櫒
def build_dataloader(data_root, resolution, batch_size, num_workers, shuffle=True):
    # 绠€鍗?ImageFolder 鏁版嵁鍔犺浇鍣ㄧ敤浜庢棤鏂囨湰璁粌
    tfm = transforms.Compose([
        # BILINEAR涓虹缉鏀惧浘鐗囨椂浣跨敤鍙岀嚎鎬ф彃鍊?
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    # .ImageFolder()閬嶅巻璺緞涓嬬殑鍥剧墖骞惰褰曡矾寰勶紝璇诲彇鏃剁粡杩噒rnasform澶勭悊
    # 浣嗗繀椤绘敞鎰忥紝鍥剧墖骞朵笉鐩存帴鏀惧湪data_root涓嬭€屾槸鏀惧湪鍏跺瓙鏂囦欢澶逛腑
    # ImageFolder鎵弿 data_root 涓嬬殑涓€绾у瓙鏂囦欢澶瑰悕浣滀负绫诲埆鍚嶏紝杩斿洖tuple(image, class_idx)
    ds = datasets.ImageFolder(data_root, transform=tfm)
    # num_workers鎸囧畾绾跨▼姹犳暟
    # pin_memory=True鐢ㄤ簬灏嗘暟鎹攣瀹氬湪鍐呭瓨涓笉鎹㈤〉鑷崇‖鐩橈紝鐢ㄤ簬鎻愰€?
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# 鏂囨湰+鍥剧墖璁粌鐨勬暟鎹姞杞藉櫒
def build_jsonl_dataloader(jsonl_path, img_root, resolution, batch_size, num_workers, shuffle=True):
    # jsonl 鏁版嵁闆嗗彲鍚屾椂鎻愪緵鍥惧儚涓庢枃鏈?caption锛岀敤浜庢潯浠惰缁?
    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    # 鑷畾涔塂ataset绫诲瀷瀹炰緥
    ds = TextImageJsonl(jsonl_path, img_root=img_root, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def diffusion_loss(model, vae, schedule: DiffusionSchedule, x0, context=None):
    # 涓€杞缁冪殑鎹熷け璁＄畻锛歎Net 棰勬祴鍣０锛孧SE 涓轰富鎹熷け锛岄檮鍔犲皬鏉冮噸鐨?posterior KL 姝ｅ垯
    device = x0.device
    # 绔埌绔缁冩椂涓嶅喕缁?VAE锛屾柟渚胯仈鍚堜紭鍖?
    # 缂栫爜寰?latent z 涓庡悗楠屽垎甯?
    z, posterior = vae.encode(x0)  
    # 鍙朾atch_size
    bsz = z.size(0)
    # t浠ｈ〃姣忎釜鏍锋湰瀵瑰簲鐨勬椂闂存[B]
    t = schedule.sample_timesteps(bsz)
    noise = torch.randn_like(z)  # 鐪熷疄鍣０

    # x_t = sqrt(alpha_bar) * z + sqrt(1-alpha_bar) * noise
    # alphas_cumprod褰㈢姸涓篬B]锛岄渶鎵╁睍鑷砙B,1,1,1]
    alpha_bar = schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
    # z鍔犲櫔鍚?
    noisy_z = torch.sqrt(alpha_bar) * z + torch.sqrt(1 - alpha_bar) * noise

    # UNet 棰勬祴鍣０锛宲red渚濇棫鏄痆B,4,32,32]
    pred = model(noisy_z, t, context)  
    mse = F.mse_loss(pred, noise, reduction="mean")

    # posterior KL 浣滀负杈呭姪姝ｅ垯椤癸紙鏉冮噸寰堝皬锛?
    kl = posterior.kl().mean()
    return mse + 1e-4 * kl, {"mse": mse.item(), "kl": kl.item()}


@torch.no_grad()
def evaluate(unet, vae, schedule, dataloader, device, max_batches=50, text_encoder=None, tokenizer=None, caption_drop=0.0):
    # 鍦ㄩ獙璇侀泦涓婅绠楀钩鍧?loss/mse/kl锛屼笉鍥炰紶姊害銆?
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

    # 浼樺厛鍔犺浇 jsonl锛堝甫 caption锛夛紝鍚﹀垯浣跨敤 ImageFolder
    if args.jsonl is not None:
        dataloader = build_jsonl_dataloader(args.jsonl, args.img_root, args.resolution, args.batch_size, args.num_workers, shuffle=True)
    else:
        dataloader = build_dataloader(args.data_root, args.resolution, args.batch_size, args.num_workers, shuffle=True)

    # 楠岃瘉闆嗭紙鍙€夛級
    val_loader = None
    if args.val_jsonl is not None:
        val_loader = build_jsonl_dataloader(args.val_jsonl, args.val_img_root, args.resolution, args.batch_size, args.num_workers, shuffle=False)
    elif args.val_root is not None:
        val_loader = build_dataloader(args.val_root, args.resolution, args.batch_size, args.num_workers, shuffle=False)

    vae = AutoencoderKL().to(device)
    if args.vae_ckpt:
        # 允许加载外部预训练 VAE，strict=False 以兼容部分键名不一致
        ckpt = torch.load(args.vae_ckpt, map_location=device)
        load_info = vae.load_state_dict(ckpt, strict=False)
        print(f"[vae] loaded from {args.vae_ckpt}, missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}")
        if args.freeze_vae:
            for p in vae.parameters():
                p.requires_grad = False
            print("[vae] parameters frozen (只训练 UNet)")

    unet = ModernDiffusionUNet().to(device)

    # 鏂囨湰缂栫爜鍣紙鍙€夛級鐢ㄤ簬灏?caption 缂栫爜涓?context
    text_encoder = None
    tokenizer = None
    if args.jsonl is not None:
        text_encoder, tokenizer = load_clip(device=device)

    # 瀹炰緥鍖栧姞鍣柟寮?
    schedule = DiffusionSchedule(args.num_steps, args.beta_start, args.beta_end, device)

    # 鍚屾椂璁粌unet鍜寁ae鐨勫弬鏁帮紱鑻ュ喕缁?VAE 鍒欏彧浼樺寲 UNet
    if args.freeze_vae:
        params = list(unet.parameters())
    else:
        params = list(unet.parameters()) + list(vae.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # 娣峰悎绮惧害浼樺寲锛実radient scaling
    # 浼樺寲鎴恌loat16鑳藉噺灏戞樉瀛樺崰鐢ㄤ絾浼氬鑷寸簿搴︽崯澶憋紝闇€瑕乻cale閬垮厤鎹熷け锛岃瑙佺埗绫伙細
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
    # 鍒濆鍖栨渶澶х敤浜庨獙璇侀泦娴嬭瘯姣旇緝
    best_val = float("inf")
    for epoch in range(args.epochs):
        for batch in dataloader:
            # 鑻ュ寘鍚枃瀛楀垯batch涓哄瓧鍏哥被鍨?
            if args.jsonl is not None:
                # 鑾峰彇鍥剧墖鍜屾枃瀛?
                x = batch["pixel_values"].to(device)
                captions = batch["caption"]
                # classifier-free guidance 鐨勮缁冿細闅忔満涓㈠純閮ㄥ垎鏂囨湰锛屼娇妯″瀷瀛︿細鏃犳潯浠剁敓鎴?
                if text_encoder is not None:
                    text_feat = encode_text(text_encoder, tokenizer, captions, device=device)
                    # drop_mask寰楀埌[B]鐨刐0,1,0,1,...]甯冨皵搴忓垪
                    # 瀵?batch 閲屾瘡涓牱鏈嫭绔嬮噰鏍蜂竴涓殢鏈烘暟锛岃嫢灏忎簬 caption_drop 灏辩疆涓?True
                    #  True 琛ㄧず杩欎釜鏍锋湰瑕佷涪鏂囨湰鏉′欢
                    drop_mask = torch.rand(len(captions), device=device) < args.caption_drop
                    # ~鍙栧弽锛屽皢鎺╃爜浣滅敤浜庢枃鏈壒寰佸疄鐜伴殢鏈轰涪寮冿紝骞舵墿灞曟垚[B,1]
                    # encode_text()杈撳嚭[B, hidden_dim]锛屾澶?[B, hidden_dim] * [B, 1](broadcast)锛岄€愬厓绱犵浉涔?
                    text_feat = text_feat * ((~drop_mask).float().unsqueeze(1))
                else:
                    text_feat = None
                context = text_feat
            # 涓嶅惈鏂囨湰淇℃伅鏃朵负鍏冪粍锛岋紙image, class_idx锛?
            else:
                x, _ = batch
                x = x.to(device)
                context = None

            # 浼樺寲杩囩▼
            # 娓呯┖姊害
            optimizer.zero_grad()

            # 娣峰悎绮惧害浼樺寲
            autocast_ctx = torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda")
            with autocast_ctx:
                # loss: mse + weight*kl
                # status: string
                loss, stats = diffusion_loss(unet, vae, schedule, x, context=context)
            # scale/step/update鏄埗绫籺orch.amp.GradSclaler绫讳笅鐨勬柟娉曪紝.backward()涓簍orch.Tensor绫讳笅鐨勬柟娉?
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 璁粌杩涘害鎵撳嵃
            if global_step % args.log_interval == 0:
                print(f"step {global_step} | loss {loss.item():.4f} | mse {stats['mse']:.4f} | kl {stats['kl']:.4f}")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{global_step},train,{loss.item():.6f},{stats['mse']:.6f},{stats['kl']:.6f}\n")
            # 瀹氭淇濆瓨ckpt
            if global_step % args.ckpt_interval == 0 and global_step > 0:
                save_ckpt(args, vae, unet, optimizer, global_step)
            # 鑻ラ獙璇侀泦瀛樺湪鍒欏畾姝ユ鏌ユā鍨嬪湪楠岃瘉闆嗕笂鐨勮〃鐜?
            if val_loader is not None and global_step % args.val_interval == 0 and global_step > 0:
                # 浠呰繑鍥?loss": total_loss / n,    "mse": total_mse / n,   "kl": total_kl / n
                val_stats = evaluate(unet, vae, schedule, val_loader, device, max_batches=args.val_max_batches,
                                     text_encoder=text_encoder, tokenizer=tokenizer, caption_drop=args.caption_drop if args.jsonl else 0.0)
                if val_stats["loss"] is not None:
                    #鎵撳嵃杩斿洖鍊?
                    print(f"[val] step {global_step} | loss {val_stats['loss']:.4f} | mse {val_stats['mse']:.4f} | kl {val_stats['kl']:.4f}")
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{global_step},val,{val_stats['loss']:.6f},{val_stats['mse']:.6f},{val_stats['kl']:.6f}\n")
                    # 瀵绘壘鏈€浼樺€硷紝鑻ユ洿灏忓垯淇濆瓨璇ユā鍨?                    if val_stats["loss"] < best_val:
                        best_val = val_stats["loss"]
                        best_ckpt_path = save_ckpt(args, vae, unet, optimizer, global_step, final=False, best=True)
                        # 璁板綍鈥滃綋鍓嶆渶浣抽獙璇佺粨鏋溾€濆埌鐙珛琛紝渚夸簬鍚庣画绛涢€?best step 涓?best ckpt
                        with best_log_path.open("a", encoding="utf-8") as f:
                            f.write(f"{global_step},{val_stats['loss']:.6f},{val_stats['mse']:.6f},{val_stats['kl']:.6f},{best_ckpt_path}\n")
                        # 鏈€浣抽獙璇侀泦琛ㄧ幇鏇存柊浜庣鍑犳
                        print(f"[ckpt] best val updated at step {global_step}")
            global_step += 1

            # 闃叉瓒婄晫
            if global_step >= args.max_steps:
                break
        # 闃叉瓒婄晫
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
    }, path)
    print(f"[ckpt] saved to {path}")
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None, help="ImageFolder root for unconditional training")
    p.add_argument("--jsonl", type=str, default=None, help="jsonl annotation path for text-image training")
    p.add_argument("--img_root", type=str, default=None, help="base image root used by jsonl image paths")
    p.add_argument("--val_root", type=str, default=None, help="validation ImageFolder root")
    p.add_argument("--val_jsonl", type=str, default=None, help="validation jsonl annotation path")
    p.add_argument("--val_img_root", type=str, default=None, help="validation image root used by jsonl paths")
    p.add_argument("--val_interval", type=int, default=1000, help="run validation every N steps")
    p.add_argument("--val_max_batches", type=int, default=50, help="max validation batches per eval")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--num_steps", type=int, default=1000, help="diffusion timesteps T")
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", help="enable AMP mixed precision")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--caption_drop", type=float, default=0.1, help="text dropout rate for classifier-free guidance")
    p.add_argument("--vae_ckpt", type=str, default=None, help="external VAE state_dict path")
    p.add_argument("--freeze_vae", action="store_true", help="freeze VAE and train UNet only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


