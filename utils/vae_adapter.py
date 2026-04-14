import torch
import torch.nn as nn

from models import AutoencoderKL as ToyAutoencoderKL


class DiffusersVAEAdapter(nn.Module):
    # Adapter that exposes the same interface as the toy AutoencoderKL.

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.scaling_factor = float(getattr(model.config, "scaling_factor", 0.18215))

    def encode(self, x):
        posterior = self.model.encode(x).latent_dist
        z = posterior.sample() * self.scaling_factor
        return z, posterior

    def decode(self, z):
        z = z / self.scaling_factor
        return self.model.decode(z).sample

    def forward(self, x):
        z, posterior = self.encode(x)
        rec = self.decode(z)
        return rec, posterior

    # Save/load with raw diffusers keys (no "model." prefix).
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)


def extract_vae_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "vae" in ckpt_obj:
        return ckpt_obj["vae"]
    return ckpt_obj


def create_vae(backend, device, source=None):
    if backend == "toy":
        return ToyAutoencoderKL().to(device)
    if backend == "diffusers":
        from diffusers import AutoencoderKL as DiffusersAutoencoderKL

        model_id = source or "stabilityai/sd-vae-ft-mse"
        model = DiffusersAutoencoderKL.from_pretrained(model_id).to(device)
        return DiffusersVAEAdapter(model).to(device)
    raise ValueError(f"Unknown VAE backend: {backend}")

