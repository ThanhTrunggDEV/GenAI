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
except ImportError:
    import sys
    sys.path.append(".")
    from motif.models.combine_embeddings import CombinedEmbedding

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--output_dir", type=str, default="outputs/hmong-pattern-lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
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
        self.embeddings_dir = self.data_root / "embeddings" # Assumed pre-computed
        
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

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.detach().item()}")
                
    # Save
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
