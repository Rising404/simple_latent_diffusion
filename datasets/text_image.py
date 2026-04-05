import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


class TextImageJsonl(Dataset):
    """
    简单的 jsonl 数据集，每行包含:
    {"image": "relative/or/absolute/path.jpg", "caption": "text ..."}
    """

    def __init__(self, jsonl_path, img_root=None, transform=None):
        self.items = []
        self.img_root = Path(img_root) if img_root else None
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(obj)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = Path(rec["image"])
        if self.img_root and not img_path.is_absolute():
            img_path = self.img_root / img_path
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        caption = rec.get("caption", "")
        return {"pixel_values": img, "caption": caption}
