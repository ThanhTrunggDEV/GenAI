#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Hmong Pattern Generation
Fine-tunes Stable Diffusion using LoRA with Cultural & Visual Conditioning
"""

import argparse
import logging
import math
import os

# Fix for OMP: Error #15 on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Import local modules
# For Kaggle/Colab, we might need to adjust python path or install the package
try:
    from motif.models.combine_embeddings import CombinedEmbedding
    from motif.models.losses import CombinedLoss
    from motif.models.cultural_encoder import CulturalSemanticEncoder
except ImportError:
    import sys
    sys.path.append(".")
    from motif.models.combine_embeddings import CombinedEmbedding
    from motif.models.losses import CombinedLoss
    from motif.models.cultural_encoder import CulturalSemanticEncoder

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="outputs/hmong-pattern-lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--data_dir", type=str, default="dataset/training/train")
    
    args = parser.parse_args()
    return args

class HmongPatternDataset(Dataset):
    """Dataset for training with image-text pairs + cultural embeddings"""
    def __init__(self, data_root, tokenizer, size=512):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.size = size
        
        # Lists
        self.images = sorted(list((self.data_root / "images").glob("*.jpg")))
        self.captions_dir = self.data_root / "captions"
        
        # Load cultural embeddings from .npz
        # Assuming embeddings are in dataset/embeddings/cultural_embeddings.npz relative to workspace
        # or in data_root/embeddings/cultural_embeddings.npz
        emb_path = self.data_root.parent.parent / "embeddings" / "cultural_embeddings.npz"
        if not emb_path.exists():
            emb_path = Path("dataset/embeddings/cultural_embeddings.npz")
            
        self.cultural_embeddings = {}
        if emb_path.exists():
            try:
                data = np.load(emb_path)
                filenames = data['filenames']
                embeddings = data['embeddings']
                # Create map: stem -> embedding
                for f, emb in zip(filenames, embeddings):
                    stem = Path(f).stem
                    self.cultural_embeddings[stem] = torch.from_numpy(emb)
                print(f"Loaded {len(self.cultural_embeddings)} cultural embeddings from {emb_path}")
            except Exception as e:
                print(f"Error loading cultural embeddings: {e}")
        else:
            print(f"Warning: Cultural embeddings not found at {emb_path}")
        
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        example = {}
        image_path = self.images[i]
        
        # Load Image
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["pixel_values"] = self.transforms(image)
        
        # Load Caption
        caption_path = self.captions_dir / f"{image_path.stem}.txt"
        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            caption = "Hmong traditional pattern" # Fallback
            
        example["input_ids"] = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        
        # Internal cultural embeddings
        if image_path.stem in self.cultural_embeddings:
            example["cultural_embeddings"] = self.cultural_embeddings[image_path.stem]
        else:
            # Placeholder/Zero embedding if missing (dim=256 default)
            example["cultural_embeddings"] = torch.zeros(256)
        
        return example

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze pre-trained models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Enable gradient checkpointing to save memory
    unet.enable_gradient_checkpointing()
    
    # Enable VAE slicing and tiling for lower memory usage
    vae.enable_slicing()
    vae.enable_tiling()

    # Use LoRA for memory efficiency (Fixes OOM on 12GB cards)
    unet.requires_grad_(False)
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    
    dataset = HmongPatternDataset(args.data_dir, tokenizer, size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1
    )
    
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Initialize Cultural Loss
    # We initialize it with a dummy encoder or just default settings
    # The loss will internally create the visual encoder
    cultural_encoder = CulturalSemanticEncoder(embedding_dim=256)
    cultural_loss_fn = CombinedLoss(
        cultural_encoder=cultural_encoder,
        lambda_diffusion=1.0,
        lambda_cultural=0.3, # Adjust weight as needed
        lambda_color=0.0 # Disable color loss for now or enable if needed
    )
    cultural_loss_fn.to(accelerator.device)
    
    # Train!
    print("***** Starting training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                steps = timesteps.long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, steps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                model_pred = unet(noisy_latents, steps, encoder_hidden_states).sample

                # -------------------
                # Compute Loss
                # -------------------
                
                # 1. Decode latents to get approximated images (for cultural loss)
                # OPTIMIZATION: Decode only a subset of the batch to save memory
                # or only decode every N steps
                
                # Only compute cultural loss for the first item in the batch to save memory
                # This significantly reduces VRAM usage while still providing guidance
                
                with torch.no_grad():
                    alpha_prod_t = noise_scheduler.alphas_cumprod[steps]
                    beta_prod_t = 1 - alpha_prod_t
                    
                    # Reshape alphas for broadcasting (B, 1, 1, 1)
                    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                    beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                    
                    # Predict original latents (x_0)
                    pred_original_latents = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                    pred_original_latents = pred_original_latents / vae.config.scaling_factor
                
                # Slicing: Only decode first N images if batch is large
                # For batch_size=4, maybe only decode 1 or 2
                max_decode_batch = 1 
                
                pred_images_subset = vae.decode(pred_original_latents[:max_decode_batch].to(dtype=vae.dtype)).sample
                
                # Get targets subset
                target_metadata = batch.get("cultural_embeddings", None)
                if target_metadata is not None:
                    target_metadata_subset = target_metadata[:max_decode_batch]
                else:
                    target_metadata_subset = None
                
                # Compute combined loss
                # We need to handle the mismatch in batch size between diff loss and cultural loss
                loss_diffusion = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                loss_cultural = torch.tensor(0.0, device=accelerator.device)
                
                if target_metadata_subset is not None:
                     # Get corresponding real images for this subset
                     target_images_subset = batch["pixel_values"][:max_decode_batch]
                     
                     # Calculate cultural loss (now acts as perceptual/feature consistency loss)
                     # We pass both metadata AND target images
                     loss_cultural = cultural_loss_fn.cultural_loss(
                         pred_images_subset, 
                         target_metadata=target_metadata_subset,
                         target_images=target_images_subset
                     )
                
                total_loss = loss_diffusion + 0.3 * loss_cultural
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {total_loss.detach().item()}")
                print(f"   Diff: {loss_diffusion.item():.4f}, Cultural: {loss_cultural.item():.4f}")
                
    # Save LoRA weights only
    if accelerator.is_main_process:
        # Unwrap the model to get the PeftModel
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # Save only the LoRA adapters
        unwrapped_unet.save_pretrained(args.output_dir)
        
        print(f"LoRA adapters saved to {args.output_dir}")

if __name__ == "__main__":
    main()
