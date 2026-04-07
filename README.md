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

让我梳理一下我们如何构建的网络框架：

【宏观网络架构】：

整体上，我们按照传统U-Net的架构搭建的网络，左侧下采样提取图像特征，谷底引入残差+注意力+残差处理特征，右侧上采样+左侧保留的skip还原图像。

我们的主要数据流包括：
1. 图像x： [Batch_size, channels, height, width]
2. 文本context： [batch_size, seq_len, context_dim]
3. 时间步timesteps序列：[batch_size]

图像往往可以直接输入，但:
 - context理论上是经过tokenizer、embedding得到的，和图像x进行处理时需要转化格式，即 [B, L, context_dim] 和 [B, H*W, channels] 对齐，当然中间肯定需要Linear(context_dim, channels)，最终得到的注意力矩阵为[B, H*W, seq_len] 
 - 时间步序列也需要经过time_embedding得到对应的向量含义[ batch_size, time_emb_dim, (broadcast),  (broadcast) ]
对于下中上采样快，上述三要素即为各个块的输入。

channels上:
 - 下采样块: 四次，扩大四次特征范围，从channels扩大到channels*4
 - 中采样块: channels不做变化
 - 上采样块: 四次，压缩四次特征范围，但不同的是，上采样块每个块的输入为in_channels + skip_channels，skip为下采样过程保留的原件，输入时torch.cat([x,skip], dim=1)用于辅助上采样恢复

每个采样块都重点依赖于ResnetBlock，AttentionBlock(SelfAttention & CrossAttention)：
 - 下采样块： （残差+注意）+（残差+注意）+ 下采样   // x, context, timesteps
 - 中采样块：  残差 + 注意 + 残差                 // x, context, timesteps
 - 上采样块： （残差+注意）+（残差+注意）+ 上采样   //x+skip, context, timesteps

【零件实现】：

此处仅讲网络部分，暂时忽略处理context原始序列的部件（网络外）。网络内使用的零件包括：

 - time_embedding函数和类: 将 timesteps[B] 映射到 time_emb [B, time_emb_dim]

 - ResnetsBlock:
    输入为x, time_emb
    1. h = _Convv2d(_SiLU(_GroupNorm(x)))
    2. h += time_emb[:,:, None, None]   //broadcast广播时间步到各个通道，而后在H和W上相加
    3. h = _Convv2d(_SiLU(_GroupNorm(h)))
    4. return h + short_cut(x)      //short_cut当in_channels!=outchannels时用Conv2d将x与h的channels对齐，而后在H和W上相加

 - AttentionBlock ( SelfAttention + CrossAttention + ffn(Linear+act+Linear) )
    输入为x, context, num_heads
    1. h = _GroupNorm(x)
    // 此处真实定义的SelfAttention内部包含了 return x+out，所以SelfAttention本身学习的就是残差，无需+=
    2. h = _SelfAttention(h)
    // 但我写CrossAttention的时候忘了内化残差了，所以需要+=
    3. h += _CrossAttention(h, context)
    // ffn直接在AttentionBlock里定义的，也没有包含残差，故+=
    4. h += ffn(h)
    4. return x + h      //attention学习的也是残差h