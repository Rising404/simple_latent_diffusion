"""
CLIP 文本编码封装，便于训练/推理共用。
依赖 open_clip_torch：pip install open_clip_torch
"""
import torch
import open_clip


def load_clip(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    # 只用文本分支，图像分支可忽略
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


@torch.no_grad()
def encode_text(model, tokenizer, prompts, device="cpu"):
    """
    prompts: list[str]
    return: torch.Tensor [B, hidden_dim] 已做 L2 归一化
    """
    tokens = tokenizer(prompts).to(device)
    text_feat = model.encode_text(tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat
