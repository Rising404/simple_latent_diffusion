# Latent Diffusion Model 

这是一个精简后的扩散模型项目，包含训练与采样推理所需核心代码。

## 目录结构

- `train.py`：训练入口
- `sampler.py`：推理/采样入口（支持 prompt 和图生图参数）
- `models/`：模型定义（Attention-based UNet + VAE）
- `datasets/`：数据集读取
- `utils/`：文本编码等工具

## 环境建议

建议使用 conda/venv 环境，并安装项目所需依赖（如 `torch`、`torchvision`、`Pillow`、文本编码相关包）。

## 训练示例

```bash
python train.py --data_root /path/to/images --resolution 256 --batch_size 4 --max_steps 10000
```

使用带文本标注数据（jsonl）训练：

```bash
python train.py --jsonl /path/to/train.jsonl --img_root /path/to/images --resolution 256 --batch_size 4
```

## 推理示例

```bash
python sampler.py --ckpt runs/ckpt_final.pt --n 4 --steps 50 --eta 0.0 --guidance 1.0 --out samples.png
```

