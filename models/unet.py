import torch  # 张量与自动求导
import torch.nn as nn  # 模块与层


# --- Time embedding utilities ---
# timesteps为输入时间步序列
# dim表示经过编码后，每个时间步输出多少维度的信息
# max_period控制频率范围，仅服务于embedding编码方式，与timesteps，dim无关
def timestep_embedding(timesteps, dim, max_period=10000):
    device = timesteps.device  

    # sin/cos 各占一半，若dim为奇数后补充
    half = dim // 2  

    # 编码每个时间步在预计的输出维度上的频率关系
    # 方法同transformer经典论文位置编码
    # 得到freqs[]，长half
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=device))
        * torch.arange(0, half, device=device).float()
        / half
    )  

    # timesteps为[B]，[:, None]扩展成[B, 1]
    # freqs[None]扩展为[1, half]
    # 广播相乘得到[B, half]
    args = timesteps.float()[:, None] * freqs[None] 

    # 拼出 [B, dim 或 dim-1]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  

    # dim为奇数则在[B, dim-1]末尾补一行全0[B, 1]，得到[B, dim]
    # 补0是位置编码方法规定的，主要用于保证输出维度匹配
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) 

    # [B, dim]
    return emb  



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),  # 线性升维
            nn.SiLU(),               # 非线性
            nn.Linear(dim * 4, dim), # 回到原维
        )

    def forward(self, t):
        emb = timestep_embedding(t, self.mlp[0].in_features)  
        return self.mlp(emb)  



class IndustrialUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # 最近邻上采样
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 3x3 卷积

    def forward(self, x):
        return self.conv(self.upsample(x))  # 上采样后卷积


# --- Core blocks ---
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)              # 归一化
        self.act1 = nn.SiLU()                                   # 激活
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)  # 3x3 卷积

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )  # 将时间嵌入映射到通道数

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )  # 通道不一致时用 1x1 对齐

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))               # 第一层
        t = self.time_emb_proj(t_emb)[:, :, None, None]        # 时间嵌入加维
        h = h + t                                              # 融合时间
        h = self.conv2(self.act2(self.norm2(h)))               # 第二层
        return h + self.shortcut(x)                            # 残差输出


class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.norm = nn.GroupNorm(32, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)  # 生成 QKV
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)    # 输出映射

    def forward(self, x):
        N, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        q = q.view(N, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [N, heads, HW, dim]
        k = k.view(N, self.num_heads, self.head_dim, H * W)                      # [N, heads, dim, HW]
        v = v.view(N, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [N, heads, HW, dim]
        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)                      # 缩放点积
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                             # 加权求和
        out = out.permute(0, 1, 3, 2).contiguous().view(N, C, H, W)             # 还原形状
        out = self.proj_out(out)
        return out + x                                                          # 残差


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        # self时使用的conv2d，此处使用的linear，更多的取决去输入格式
        # self输入为[N, C, H, W]，cross输入为[N, seq_len, context_dim]
        # 前者C在HW两维上，后者在seq_len上，决定了前者适合conv2d，后者适合linear
        # nn.Linear封装自动把输入张量的最后一维作为要处理的in_features
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.proj_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        # 输入图像x格式变化，本质N, H*W, C
        N, L, C = x.shape
        q = self.to_q(x).view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, HW, dim]

        # context输入格式N, seq_len, context_dim，seq_len其实就是文本长、token数
        # attn其实是在求 [H*W, seq_len] 的注意力分数
        # 此处-1其实就是seq_len，-1为pytorch语法自动推断维度，可灵活适配
        # pytorch自动推理基于当前张量实际形状，总元素已知、其它维已知，可求
        k = self.to_k(context).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, heads, dim, seq]
        v = self.to_v(context).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, seq, dim]
        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(N, L, C)
        return self.proj_out(out)


class AttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.norm_in = nn.GroupNorm(32, query_dim)
        # 图像多头自注意力，获取图片不同位置之间的关系
        self.attn1 = SelfAttention(query_dim, num_heads=num_heads)
        # 图像流中以图像质询文字，获取图文关系，依旧输出图像信息
        self.attn2 = CrossAttention(query_dim=query_dim, context_dim=context_dim, num_heads=num_heads)
        # 注意力机制只能获取位置与位置、位置与文字之间的联系
        # 和transformer经典架构一样，引入前馈网络进一步增强特征表达能力
        hidden_dim = query_dim * 4
        # ffn升维度、激活、降维
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            # 这里使用GELU激活，原论文使用SiLU，两者性能相近，GELU在transformer中更常见
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        h = self.norm_in(x)
        # 此处写的不是很统一，可能稍稍有点混乱
        # attn1调用的SelfAttention中已经封装了残差，故此处直接调用
        h = self.attn1(h)  
        h_seq = h.view(B, C, H * W).permute(0, 2, 1)  # 展平 [B, HW, C]
        if context is not None:
            # attn2调用的CrossAttention中没有封装残差，故此处手动加上
            h_seq = h_seq + self.attn2(h_seq, context)  # 交叉注意力
        # 同理，本块中的ffn也没有封装残差，手动加上
        h_seq = h_seq + self.ffn(h_seq)
        h = h_seq.permute(0, 2, 1).contiguous().view(B, C, H, W)  # 还原
        # attention学习的也是残差
        return x + h 


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim, num_layers=2):
        super().__init__()
        self.resnets = nn.ModuleList()     # 多个 ResNet 堆叠
        self.attentions = nn.ModuleList()  # 对应的注意力
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(in_channels=in_c, out_channels=out_channels, time_emb_dim=time_emb_dim)
            )
            self.attentions.append(AttentionBlock(query_dim=out_channels, context_dim=context_dim))
        self.downsampler = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)  # 下采样

    def forward(self, x, t_emb, context=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t_emb)
            x = attn(x, context)
        x = self.downsampler(x)          # 空间减半
        skip = x                         # 保存 skip（同尺寸）
        return x, skip


class MidBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, context_dim):
        super().__init__()
        self.res1 = ResnetBlock(channels, channels, time_emb_dim)  # Res
        self.attn = AttentionBlock(channels, context_dim)          # Self/Cross Attn
        self.res2 = ResnetBlock(channels, channels, time_emb_dim)  # Res

    def forward(self, x, t_emb, context=None):
        x = self.res1(x, t_emb)
        x = self.attn(x, context)
        x = self.res2(x, t_emb)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, context_dim, num_layers=2):
        super().__init__()
        self.resnets = nn.ModuleList()     # 堆叠层
        self.attentions = nn.ModuleList()  # 对应注意力
        for i in range(num_layers):
            # 上采样过程中输入为当前层输入与 skip 连接的拼接
            res_in_c = (in_channels + skip_channels) if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(in_channels=res_in_c, out_channels=out_channels, time_emb_dim=time_emb_dim)
            )
            self.attentions.append(AttentionBlock(query_dim=out_channels, context_dim=context_dim))
        self.upsampler = IndustrialUpConv(in_channels=out_channels, out_channels=out_channels)  # 上采样

    def forward(self, x, skip, t_emb, context=None):
        x = torch.cat([x, skip], dim=1)  # 拼接 skip
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t_emb)
            x = attn(x, context)
        x = self.upsampler(x)  # 分辨率翻倍
        return x


class ModernDiffusionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=320, time_emb_dim=1280, context_dim=768):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)                       # 时间嵌入
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)  # 输入映射

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(base_channels, base_channels, time_emb_dim, context_dim),
                DownBlock(base_channels, base_channels * 2, time_emb_dim, context_dim),
                DownBlock(base_channels * 2, base_channels * 4, time_emb_dim, context_dim),
                DownBlock(base_channels * 4, base_channels * 4, time_emb_dim, context_dim),
            ]
        )

        self.mid_block = MidBlock(base_channels * 4, time_emb_dim, context_dim)  # 谷底

        self.up_blocks = nn.ModuleList(
            [
                UpBlock(base_channels * 4, base_channels * 4, base_channels * 4, time_emb_dim, context_dim),
                UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim, context_dim),
                UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim, context_dim),
                UpBlock(base_channels, base_channels, base_channels, time_emb_dim, context_dim),
            ]
        )

        self.norm_out = nn.GroupNorm(32, base_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)  # 输出映射

    def forward(self, x, time, context=None):
        t_emb = self.time_mlp(time)  # 时间编码
        h = self.conv_in(x)          # 输入映射
        skips = []
        for block in self.down_blocks:   # 逐层下采样
            h, skip = block(h, t_emb, context)
            skips.append(skip)

        h = self.mid_block(h, t_emb, context)

        for block in self.up_blocks:     # 逐层上采样
            skip = skips.pop()
            h = block(h, skip, t_emb, context)

        h = self.norm_out(h)
        h = self.act_out(h)
        return self.conv_out(h)
