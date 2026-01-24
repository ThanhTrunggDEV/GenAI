#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Loss Functions for Culturally-Constrained Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


from .visual_encoder import HmongVisualEncoder

class CulturalConsistencyLoss(nn.Module):
    """
    Loss to enforce cultural consistency between generated and real patterns
    Compares cultural embeddings
    """
    
    def __init__(self, cultural_encoder, visual_encoder=None):
        super().__init__()
        self.cultural_encoder = cultural_encoder
        
        # Use provided visual encoder or create new one matching cultural dim
        if visual_encoder:
            self.visual_encoder = visual_encoder
        else:
            # Initialize with same embedding dim as cultural encoder
            # Default cultural dim is 256
            emb_dim = getattr(cultural_encoder, 'embedding_dim', 256)
            self.visual_encoder = HmongVisualEncoder(embedding_dim=emb_dim, pretrained=True)
            
        # Freeze visual encoder parameters to avoid "chasing a moving target"
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(self, generated_images, target_metadata):
        """
        Args:
            generated_images: Generated pattern images (B, C, H, W). Assumed to be in [-1, 1] range.
            target_metadata: Target cultural embeddings (B, dim)
        Returns:
            loss: Cultural consistency loss
        """
        # 1. Denormalize from [-1, 1] to [0, 1] (assuming diffusion output is [-1, 1])
        # Clone to avoid modifying original tensor if used elsewhere
        images = (generated_images.clone() + 1.0) * 0.5
        images = torch.clamp(images, 0.0, 1.0) # Ensure within bounds
        
        # 2. Resize images for visual encoder (224x224)
        if images.shape[-2:] != (224, 224):
            images_resized = F.interpolate(
                images, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            images_resized = images
        
        # 3. Normalize for ImageNet backbone (ResNet expectation)
        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        
        images_normalized = (images_resized - mean) / std
            
        # 4. Extract visual embeddings
        # We assume generated_images has gradients enabling backprop
        self.visual_encoder.eval()
        visual_emb = self.visual_encoder(images_normalized)
        
        # 5. Get target embeddings 
        # Assume target_metadata is already the tensor embedding
        target_emb = target_metadata
            
        # 6. Compute Cosine Similarity Loss
        # We want vectors to be similar (target=1)
        target = torch.ones(visual_emb.shape[0], device=visual_emb.device)
        loss = self.cosine_loss(visual_emb, target_emb, target)
        
        return loss


class ColorPaletteLoss(nn.Module):
    """
    Loss to ensure generated patterns match target color palette
    """
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def extract_color_histogram(self, images, bins=256):
        """
        Extract color histogram from images
        
        Args:
            images: Tensor of shape (B, C, H, W)
            bins: Number of histogram bins
        Returns:
            histograms: Tensor of shape (B, C, bins)
        """
        batch_size, channels = images.shape[0], images.shape[1]
        histograms = []
        
        for i in range(batch_size):
            img_hists = []
            for c in range(channels):
                channel = images[i, c].flatten()
                hist = torch.histc(channel, bins=bins, min=0.0, max=1.0)
                hist = hist / hist.sum()  # Normalize
                img_hists.append(hist)
            histograms.append(torch.stack(img_hists))
        
        return torch.stack(histograms)
    
    def forward(self, generated_images, target_images):
        """
        Args:
            generated_images: Generated images (B, C, H, W)
            target_images: Target images (B, C, H, W)
        Returns:
            loss: Color palette matching loss
        """
        gen_hist = self.extract_color_histogram(generated_images)
        target_hist = self.extract_color_histogram(target_images)
        
        loss = self.l1_loss(gen_hist, target_hist)
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for culturally-constrained generation
    L_total = λ₁·L_diffusion + λ₂·L_cultural + λ₃·L_color
    """
    
    def __init__(self, cultural_encoder=None, 
                 lambda_diffusion=1.0, lambda_cultural=0.3, lambda_color=0.2):
        super().__init__()
        
        self.lambda_diffusion = lambda_diffusion
        self.lambda_cultural = lambda_cultural
        self.lambda_color = lambda_color
        
        # Loss components
        self.mse_loss = nn.MSELoss()
        self.cultural_loss = CulturalConsistencyLoss(cultural_encoder) if cultural_encoder else None
        self.color_loss = ColorPaletteLoss()
    
    def forward(self, predicted_noise, actual_noise, generated_images=None, 
                target_images=None, target_metadata=None):
        """
        Compute combined loss
        
        Args:
            predicted_noise: Predicted noise from diffusion model
            actual_noise: Actual noise added
            generated_images: Generated images (optional)
            target_images: Target images (optional)
            target_metadata: Target cultural metadata (optional)
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Standard diffusion loss (MSE between predicted and actual noise)
        loss_diffusion = self.mse_loss(predicted_noise, actual_noise)
        
        total_loss = self.lambda_diffusion * loss_diffusion
        loss_dict = {'diffusion': loss_diffusion.item()}
        
        # Cultural consistency loss (if applicable)
        if self.cultural_loss is not None and generated_images is not None and target_metadata is not None:
            loss_cultural = self.cultural_loss(generated_images, target_metadata)
            total_loss += self.lambda_cultural * loss_cultural
            loss_dict['cultural'] = loss_cultural.item()
        
        # Color palette loss (if applicable)
        if generated_images is not None and target_images is not None:
            loss_color = self.color_loss(generated_images, target_images)
            total_loss += self.lambda_color * loss_color
            loss_dict['color'] = loss_color.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("🧪 Testing Loss Functions...")
    
    # Test color palette loss
    color_loss_fn = ColorPaletteLoss()
    
    fake_gen = torch.rand(2, 3, 64, 64)
    fake_target = torch.rand(2, 3, 64, 64)
    
    color_loss = color_loss_fn(fake_gen, fake_target)
    print(f"✅ Color Loss Test: {color_loss.item():.4f}")
    
    # Test combined loss
    combined_loss_fn = CombinedLoss()
    
    pred_noise = torch.rand(2, 4, 64, 64)
    actual_noise = torch.rand(2, 4, 64, 64)
    
    total_loss, loss_dict = combined_loss_fn(
        pred_noise, actual_noise,
        generated_images=fake_gen,
        target_images=fake_target
    )
    
    print(f"✅ Combined Loss Test:")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")
