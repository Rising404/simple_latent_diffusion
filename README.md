# Latent Diffusion Model

Simplified latent diffusion training and inference codebase.

## Project Structure
- `train.py`: training entry
- `sampler.py`: inference entry (text-to-image and image-to-image)
- `models/`: UNet and VAE definitions
- `utils/text_image.py`: jsonl image-text dataset loader
- `utils/text_encoder.py`: CLIP text encoder wrapper

## Data Format
- ImageFolder (unconditional): `root/class_xxx/*.jpg`
- jsonl (text-image): one sample per line, for example:

```json
{"image":"train/0001.jpg","caption":"a yellow flower"}
```

## Training
Unconditional training:

```bash
python train.py --data_root data/flowers102/train --resolution 256 --batch_size 4 --max_steps 10000
```

With validation:

```bash
python train.py --data_root data/flowers102/train --val_root data/flowers102/val --val_interval 500 --val_max_batches 50
```

Text-image training:

```bash
python train.py --jsonl /path/to/train.jsonl --img_root /path/to/images --val_jsonl /path/to/val.jsonl --val_img_root /path/to/images
```

Use external pretrained VAE and train UNet only:

```bash
python train.py --data_root data/flowers102/train --vae_ckpt pretrained_models/pretrained_vae_sd.pt --freeze_vae
```

## Inference
Unconditional sampling:

```bash
python sampler.py --ckpt runs/ckpt_final.pt --n 4 --steps 50 --out samples.png
```

Text-to-image:

```bash
python sampler.py --ckpt runs/ckpt_final.pt --prompt "a red flower in sunlight" --guidance 7.5 --n 4 --out samples_text
```

Image-to-image:

```bash
python sampler.py --ckpt runs/ckpt_final.pt --init_image path/to/image.jpg --strength 0.5 --out samples_img2img
```

Force external VAE at inference:

```bash
python sampler.py --ckpt runs/ckpt_final.pt --vae_ckpt pretrained_models/pretrained_vae_sd.pt --out samples
```

## Outputs
Default output directory is `--out_dir runs`:
- `train_log.csv`: periodic train/val metrics
- `best_val_log.csv`: best validation records with ckpt path
- `ckpt_step{N}.pt`: periodic checkpoints
- `ckpt_best_step{N}.pt`: checkpoint saved when validation improves
- `ckpt_final.pt`: final checkpoint at the end of training

